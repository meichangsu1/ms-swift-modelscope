#!/usr/bin/env python
"""Check R06's Ray sampler device mapping and vLLM-Ascend initialization without loading weights."""

import argparse
import importlib.metadata
import json
import os
import socket
import subprocess
import sys
import threading
import traceback
import uuid
from pathlib import Path


def environment():
    return {
        'host': socket.gethostname(),
        'pid': os.getpid(),
        'thread': threading.current_thread().name,
        'python': sys.executable,
        'env': {
            key: os.environ.get(key)
            for key in (
                'ASCEND_VISIBLE_DEVICES', 'ASCEND_RT_VISIBLE_DEVICES', 'ASCEND_DEVICE_ID',
                'RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES', 'RAY_ADDRESS',
                'ASCEND_HOME_PATH', 'LD_PRELOAD', 'WORKER_NAME', 'LOCAL_RANK',
            )
        },
    }


def check_npu():
    result = environment()
    try:
        import torch
        import torch_npu

        result['torch'] = torch.__version__
        result['torch_npu_file'] = torch_npu.__file__
        result['count'] = torch.npu.device_count()
        result['soc'] = torch_npu.npu.get_soc_version()
        result['allocation'] = torch.ones(1, device='npu:0').cpu().tolist()
        result['ok'] = True
    except Exception:
        result['ok'] = False
        result['error'] = traceback.format_exc()
    return result


def child_check():
    # Import in a fresh process, in the same order as the failing vLLM worker. In particular,
    # do not initialize torch_npu before importing the Ascend worker and calling its SoC check.
    result = {'before_import': environment()}
    try:
        import vllm_ascend.worker.worker as worker
        from vllm_ascend.device.device_config import check_ascend_device_type

        result['worker_file'] = worker.__file__
        result['after_import'] = environment()
        check_ascend_device_type()
        result['device_check_ok'] = True
    except Exception:
        result['device_check_ok'] = False
        result['device_check_error'] = traceback.format_exc()
    result['npu'] = check_npu()
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result['device_check_ok'] and result['npu']['ok'] else 1


def run_child(timeout):
    try:
        child = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), '--child'],
            capture_output=True, text=True, timeout=timeout,
        )
        return {'returncode': child.returncode, 'stdout': child.stdout, 'stderr': child.stderr}
    except subprocess.TimeoutExpired as exc:
        return {'returncode': None, 'error': f'Child check exceeded {timeout} seconds.',
                'stdout': str(exc.stdout or ''), 'stderr': str(exc.stderr or '')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trainer-npus', type=int, default=4)
    parser.add_argument('--sampler-npus', type=int, default=4)
    parser.add_argument('--timeout', type=int, default=120, help='Timeout for the fresh child process, in seconds.')
    parser.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        return child_check()
    if min(args.trainer_npus, args.sampler_npus, args.timeout) <= 0:
        parser.error('NPU counts and timeout must be positive.')

    sys.path.insert(0, str(Path(__file__).resolve().parents[6]))
    if os.environ.get('TWINKLE_SRC'):
        sys.path.insert(0, os.environ['TWINKLE_SRC'])

    driver = check_npu()
    driver['packages'] = {}
    for name in ('torch', 'torch-npu', 'ray', 'twinkle-kit', 'vllm', 'vllm-ascend'):
        try:
            driver['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            driver['packages'][name] = None
    print(json.dumps({'driver': driver}, indent=2), flush=True)
    total = args.trainer_npus + args.sampler_npus
    if not driver['ok'] or driver.get('count', 0) < total:
        parser.error(f'This topology requires {total} usable NPUs; see the driver report above.')

    import ray
    import twinkle
    from twinkle import DeviceGroup, remote_class, remote_function
    from twinkle.infra._ray.ray_helper import RayHelper

    @remote_class()
    class SamplerProbe:

        @remote_function(dispatch='all', collect='first', lazy_collect=False)
        def read(self, timeout):
            actor = check_npu()
            child = run_child(timeout)
            return {'actor': actor, 'child': child}

    # Own an isolated local cluster: do not attach to or stop another user's Ray job.
    namespace = 'vllm-npu-probe-' + uuid.uuid4().hex[:8]
    os.environ['TWINKLE_RAY_NAMESPACE'] = namespace
    os.environ['TWINKLE_RAY_REGISTRY_NAME'] = namespace
    try:
        ray.init(address='local', namespace=namespace, include_dashboard=False, resources={'NPU': driver['count']})
        twinkle.initialize(
            mode='ray', nproc_per_node=total,
            groups=[
                DeviceGroup(name='model', ranks=list(range(args.trainer_npus)), device_type='NPU'),
                DeviceGroup(name='sampler', ranks=list(range(args.trainer_npus, total)),
                            device_type='NPU', gpus_per_worker=args.sampler_npus),
            ],
        )
        placements = RayHelper.group_info('sampler')['placements']
        print(json.dumps({'sampler_placements': [item['gpu_rank'] for item in placements],
                          'twinkle_file': twinkle.__file__}, indent=2), flush=True)
        report = SamplerProbe(remote_group='sampler').read(args.timeout)
        print(json.dumps({'sampler': report}, indent=2), flush=True)
        return 0 if report['actor']['ok'] and report['child']['returncode'] == 0 else 1
    finally:
        twinkle.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
