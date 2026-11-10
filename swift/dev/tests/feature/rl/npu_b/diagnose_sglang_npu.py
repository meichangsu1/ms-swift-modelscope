#!/usr/bin/env python
"""Probe SGLang's Engine import in a Twinkle sampler actor without loading a model."""

import argparse
import asyncio
import importlib
import os
import socket
import sys
import threading
import traceback
import uuid
from concurrent.futures import Future, TimeoutError
from pathlib import Path

import json


def snapshot():
    import torch
    import torch_npu

    result = {
        'host': socket.gethostname(),
        'pid': os.getpid(),
        'thread': threading.current_thread().name,
        'python': sys.executable,
        'torch': torch.__version__,
        'torch_npu_file': torch_npu.__file__,
        'env': {
            key: os.environ.get(key)
            for key in (
                'ASCEND_VISIBLE_DEVICES', 'ASCEND_RT_VISIBLE_DEVICES',
                'RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES',
                'ASCEND_HOME_PATH', 'LD_LIBRARY_PATH', 'WORKER_NAME', 'LOCAL_RANK',
            )
        },
    }
    try:
        result['available'] = torch.npu.is_available()
        result['count'] = torch.npu.device_count()
    except Exception:
        result['npu_error'] = traceback.format_exc()
    return result


async def inspect_engine_import():
    result = {'before_import': snapshot()}
    try:
        module = importlib.import_module('sglang.srt.entrypoints.engine')
        result['engine_module'] = module.__file__
        result['import_ok'] = True
    except Exception:
        result['import_ok'] = False
        result['import_error'] = traceback.format_exc()
    result['after_import'] = snapshot()
    return result


def inspect_background_loop(timeout):
    future = Future()

    def run():
        try:
            future.set_result(asyncio.run(inspect_engine_import()))
        except Exception:
            future.set_result({'import_ok': False, 'probe_error': traceback.format_exc()})

    # Match the sampler: import Engine while a background thread's event loop is running.
    thread = threading.Thread(target=run, name='SGLangSampler-EventLoop', daemon=True)
    thread.start()
    try:
        return future.result(timeout=timeout)
    except TimeoutError:
        return {'import_ok': False, 'probe_error': f'Engine import exceeded {timeout} seconds.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trainer-npus', type=int, default=2)
    parser.add_argument('--sampler-npus', type=int, default=2)
    parser.add_argument('--timeout', type=int, default=120)
    args = parser.parse_args()
    if min(args.trainer_npus, args.sampler_npus, args.timeout) <= 0:
        parser.error('NPU counts and timeout must be positive.')

    sys.path.insert(0, str(Path(__file__).resolve().parents[6]))
    if os.environ.get('TWINKLE_SRC'):
        sys.path.insert(0, os.environ['TWINKLE_SRC'])

    import ray
    import twinkle
    from twinkle import DeviceGroup, remote_class, remote_function

    driver = snapshot()
    driver['twinkle_file'] = twinkle.__file__
    print(json.dumps({'driver': driver}, indent=2), flush=True)
    total = args.trainer_npus + args.sampler_npus
    if not driver.get('available') or driver.get('count', 0) < total:
        parser.error(f'This topology requires {total} visible NPUs; see the driver report above.')

    @remote_class()
    class SamplerProbe:

        def __init__(self, timeout):
            self.report = {'actor_thread': snapshot(), 'background_loop': inspect_background_loop(timeout)}

        @remote_function(dispatch='all', collect='first', lazy_collect=False)
        def read(self):
            return self.report

    # Own a fresh cluster and registry; do not stop or modify an existing training cluster.
    namespace = 'sglang-npu-probe-' + uuid.uuid4().hex[:8]
    os.environ['TWINKLE_RAY_NAMESPACE'] = namespace
    os.environ['TWINKLE_RAY_REGISTRY_NAME'] = namespace
    try:
        ray.init(address='local', namespace=namespace, include_dashboard=False,
                 resources={'NPU': driver['count']})
        twinkle.initialize(
            mode='ray', nproc_per_node=total,
            groups=[
                DeviceGroup(name='model', ranks=list(range(args.trainer_npus)), device_type='NPU'),
                DeviceGroup(name='sampler', ranks=list(range(args.trainer_npus, total)),
                            device_type='NPU', gpus_per_worker=args.sampler_npus),
            ])
        probe = SamplerProbe(timeout=args.timeout, remote_group='sampler')
        report = probe.read()
        print(json.dumps({'sampler': report}, indent=2), flush=True)
    finally:
        ray.shutdown()
    return 0 if report['background_loop']['import_ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
