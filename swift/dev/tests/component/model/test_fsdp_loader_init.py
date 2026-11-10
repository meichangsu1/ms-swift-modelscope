# Copyright (c) ModelScope Contributors. All rights reserved.
"""Checkpoint-free meta construction and real CPU FSDP shard loading through a family loader."""
import os
import time
from datetime import timedelta
from unittest.mock import patch

import pytest
import torch


def _tiny_checkpoint(directory):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
                         tie_word_embeddings=True)
    torch.manual_seed(17)
    model = Qwen2ForCausalLM(config)
    model.save_pretrained(directory)
    tokenizer = Tokenizer(WordLevel({f'w{i}': i for i in range(32)}, unk_token='w0'))
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='w0', pad_token='w0').save_pretrained(directory)


def _fsdp_loader_worker(rank, directory, with_lora, ray_layout):
    import numpy as np
    import torch.distributed as dist
    from peft import LoraConfig
    from torch.distributed.tensor import DTensor

    from swift.dev.model.loader import LoaderModelInfo, get_model_loader
    from twinkle import DeviceGroup, DeviceMesh, Platform, initialize
    from twinkle.model.transformers import TransformersModel

    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE='2', LOCAL_WORLD_SIZE='2')
    if ray_layout:
        os.environ['LOCAL_RANK'] = '0'
        os.environ.pop('LOCAL_WORLD_SIZE', None)
        os.environ.pop('LOCAL_SIZE', None)
    else:
        os.environ.pop('TWINKLE_MODE', None)
    dist.init_process_group('gloo', init_method=f'file://{directory}/pg', rank=rank, world_size=2,
                            timeout=timedelta(seconds=40))
    try:
        calls = []
        base_loader = get_model_loader('qwen2_5')

        class RecordingLoader(base_loader):
            def build_model(self, *args, **kwargs):
                assert rank == 0, 'Only the local source rank may read checkpoint weights'
                calls.append('checkpoint')
                return super().build_model(*args, **kwargs)

            def build_model_from_config(self, *args, **kwargs):
                assert rank != 0
                calls.append('config')
                return super().build_model_from_config(*args, **kwargs)

            def process_model(self, model):
                calls.append('postprocess')
                # Different local values prove the source buffer is restored on the other rank.
                model.register_buffer('loader_scratch', torch.arange(3.) + rank, persistent=False)
                return super().process_model(model)

        loader = RecordingLoader(LoaderModelInfo(model_dir=directory, model_type='qwen2_5'))
        mesh = DeviceMesh(mesh=np.arange(2), mesh_dim_names=('fsdp',), device_type='cpu')
        with patch.object(Platform, 'get_local_device', return_value='cpu'), \
                patch.object(Platform, 'device_backend', return_value='gloo'):
            initialize(mode='local', global_device_mesh=mesh,
                       groups=[DeviceGroup(name='default', ranks=[0, 1], device_type='cpu')])
            if ray_layout:
                # Keep execution local, but reproduce the environment of Ray's
                # actors: each sees one device and uses LOCAL_RANK=0.
                os.environ['TWINKLE_MODE'] = 'ray'
            wrapper = TransformersModel(model_id=directory, model_loader=loader, device_mesh=mesh,
                                        strategy='native_fsdp', memory_efficient_init=True,
                                        mixed_precision='no', gradient_checkpointing=False,
                                        dtype=torch.float32, local_files_only=True)
            assert calls == [('checkpoint' if rank == 0 else 'config'), 'postprocess']
            assert all(parameter.is_meta == (rank != 0) for parameter in wrapper.model.parameters())
            assert wrapper.model.lm_head.weight is wrapper.model.model.embed_tokens.weight
            assert wrapper._default_tokenizer is not None
            if with_lora:
                torch.manual_seed(21 + rank)
                wrapper.add_adapter_to_model('default', LoraConfig(r=2, lora_alpha=4, target_modules=['q_proj']))
            if rank == 0:
                torch.save(wrapper.model.state_dict(), f'{directory}/expected.pt')
            dist.barrier()
            wrapper.model = wrapper.strategy.wrap_model(wrapper.model)[0]
            expected = torch.load(f'{directory}/expected.pt', weights_only=True)
            actual = wrapper.model.state_dict()
            assert actual.keys() == expected.keys()
            for name, tensor in actual.items():
                assert not tensor.is_meta
                if isinstance(tensor, DTensor):
                    assert tensor.to_local().numel() < tensor.numel()
                    tensor = tensor.full_tensor()
                torch.testing.assert_close(tensor.cpu(), expected[name], rtol=0, atol=0)
            base = wrapper.model.get_base_model() if with_lora else wrapper.model
            assert base.lm_head.weight is base.model.embed_tokens.weight
            torch.testing.assert_close(base.loader_scratch, torch.arange(3.))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('with_lora', [False, True])
@pytest.mark.parametrize('ray_layout', [False, True], ids=['torchrun', 'ray'])
def test_loader_meta_init_and_fsdp_broadcast(tmp_path, with_lora, ray_layout):
    _tiny_checkpoint(tmp_path)
    context = torch.multiprocessing.spawn(_fsdp_loader_worker, args=(str(tmp_path), with_lora, ray_layout),
                                          nprocs=2, join=False)
    try:
        deadline = time.monotonic() + 60
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail('FSDP loader initialization did not finish within 60 seconds')
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)


def test_qwen35_moe_loader_builds_matching_meta_model_without_checkpoint(monkeypatch):
    from accelerate import init_empty_weights
    from transformers import Qwen3_5MoeConfig, Qwen3_5MoeForConditionalGeneration

    from swift.dev.model.loader import LoaderModelInfo, get_model_loader

    config = Qwen3_5MoeConfig(
        text_config=dict(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=8, moe_intermediate_size=16,
                         num_experts=4, num_experts_per_tok=2, layer_types=['full_attention']),
        vision_config=dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=16,
                           patch_size=2, temporal_patch_size=2, spatial_merge_size=2))
    expected = Qwen3_5MoeForConditionalGeneration(config)
    loader = get_model_loader('qwen3_5_moe')(LoaderModelInfo(model_type='qwen3_5_moe'))

    def reject_checkpoint(*args, **kwargs):
        pytest.fail('Meta construction must not read checkpoint weights')

    monkeypatch.setattr(Qwen3_5MoeForConditionalGeneration, 'from_pretrained', reject_checkpoint)
    with init_empty_weights(include_buffers=False):
        actual = loader.build_model_from_config(config, processor=None, dtype=torch.float32)
    actual = loader.process_model(actual)
    assert type(actual) is type(expected)
    assert {name: tuple(tensor.shape) for name, tensor in actual.state_dict().items()} == {
        name: tuple(tensor.shape) for name, tensor in expected.state_dict().items()}
    assert all(parameter.is_meta for parameter in actual.parameters())


def test_empty_loader_requires_config_constructor():
    from types import SimpleNamespace
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from twinkle.model.transformers import TransformersModel

    wrapper = SimpleNamespace(hf_config=Qwen2Config())
    with pytest.raises(ValueError, match=r'must implement build_model_from_config\(\)'):
        TransformersModel._init_empty_model_from_config(
            wrapper, Qwen2ForCausalLM, model_loader=SimpleNamespace(), processor=None)


def test_custom_pretrained_constructor_requires_matching_config_constructor():
    from transformers import Qwen2Config

    from swift.dev.model.loader import LoaderModelInfo, ModelLoader

    class CustomLoader(ModelLoader):
        def build_model(self, *args, **kwargs):
            raise AssertionError('Checkpoint loading must not run')

    loader = CustomLoader(LoaderModelInfo())
    with pytest.raises(ValueError, match='with the same construction adaptations'):
        loader.build_model_from_config(Qwen2Config(), processor=None)
