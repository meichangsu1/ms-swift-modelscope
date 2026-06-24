# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from huggingface/transformers.
import inspect
import os
from types import FunctionType, MethodType
from typing import List, Optional, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import PeftModel
from torch.nn import CrossEntropyLoss, Module

from swift.utils import get_dist_setting, get_logger

logger = get_logger()


def _get_deepspeed_elastic_world_size():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return get_dist_setting()[2]


def _enable_load_universal(ds_config):
    if isinstance(ds_config, dict):
        checkpoint = ds_config.get('checkpoint')
        if not isinstance(checkpoint, dict):
            checkpoint = {}
            ds_config['checkpoint'] = checkpoint
        checkpoint['load_universal'] = True


def enable_deepspeed_load_universal(args, trainer=None):
    _enable_load_universal(getattr(args, 'deepspeed', None))

    hf_ds_config = getattr(args, 'hf_deepspeed_config', None)
    _enable_load_universal(getattr(hf_ds_config, 'config', None))

    deepspeed_plugin = getattr(args, 'deepspeed_plugin', None)
    if trainer is not None and deepspeed_plugin is None:
        accelerator = getattr(trainer, 'accelerator', None)
        state = getattr(accelerator, 'state', None)
        deepspeed_plugin = getattr(state, 'deepspeed_plugin', None)
    if deepspeed_plugin is not None:
        _enable_load_universal(getattr(deepspeed_plugin, 'deepspeed_config', None))
        plugin_hf_ds_config = getattr(deepspeed_plugin, 'hf_ds_config', None)
        _enable_load_universal(getattr(plugin_hf_ds_config, 'config', None))


def prepare_deepspeed_elastic_config(args, state=None):
    ds_config = getattr(args, 'deepspeed', None)
    if not ds_config:
        return
    if not isinstance(ds_config, dict):
        logger.warning('DeepSpeed elastic expects args.deepspeed to be a dict, but got '
                       f'{type(ds_config).__name__}. Skip elastic config.')
        return

    from deepspeed.elasticity import compute_elastic_config
    from deepspeed.git_version_info import version as __version__

    enable_deepspeed_load_universal(args)
    elasticity = ds_config.get('elasticity') or {}
    if not elasticity:
        logger.warning('DeepSpeed elastic callback is enabled, but no `elasticity` section is found in '
                       'the DeepSpeed config. Only `checkpoint.load_universal` is enabled.')
        return
    if elasticity.get('enabled') is False:
        return

    world_size = _get_deepspeed_elastic_world_size()
    final_batch_size, _, micro_batch_size = compute_elastic_config(
        ds_config=ds_config,
        target_deepspeed_version=__version__,
        world_size=world_size,
    )
    if world_size <= 0 or micro_batch_size <= 0:
        raise ValueError('DeepSpeed elastic config produced invalid batch settings: '
                         f'world_size={world_size}, micro_batch_size={micro_batch_size}.')
    gradient_accu_steps = max(1, final_batch_size // (micro_batch_size * world_size))
    args.per_device_train_batch_size = micro_batch_size
    args.gradient_accumulation_steps = gradient_accu_steps
    if state is not None:
        state.train_batch_size = args.per_device_train_batch_size * max(1, args.n_gpu)
    logger.info('DeepSpeed elastic config is enabled. '
                f'world_size: {world_size}, '
                f'per_device_train_batch_size: {args.per_device_train_batch_size}, '
                f'gradient_accumulation_steps: {args.gradient_accumulation_steps}')


def can_return_loss(model: Module) -> bool:
    """Check if a given model can return loss."""
    if isinstance(model, PeftModel):
        signature = inspect.signature(model.model.forward)
    else:
        signature = inspect.signature(model.forward)
    for p in signature.parameters:
        if p == 'return_loss' and signature.parameters[p].default is True:
            return True
    return False


def find_labels(model: Module) -> List[str]:
    """Find the labels used by a given model."""
    model_name = model.__class__.__name__
    if isinstance(model, PeftModel):
        signature = inspect.signature(model.model.forward)
    else:
        signature = inspect.signature(model.forward)
    if 'QuestionAnswering' in model_name:
        return [p for p in signature.parameters if 'label' in p or p in ('start_positions', 'end_positions')]
    else:
        return [p for p in signature.parameters if 'label' in p]


def get_function(method_or_function: Union[MethodType, FunctionType]) -> FunctionType:
    if isinstance(method_or_function, MethodType):
        method_or_function = method_or_function.__func__
    return method_or_function


def is_instance_of_ms_model(model: Module) -> bool:
    """avoid import modelscope: circular dependency problem"""
    for m_cls in model.__class__.__mro__:
        cls_name = m_cls.__name__
        cls_module = m_cls.__module__
        if cls_name == 'Model' and cls_module.startswith('modelscope'):
            return True
    return False


def per_token_loss_func_sp(outputs, labels, enable_dft_loss=False, **kwargs) -> torch.Tensor:
    """Common loss function for sequence parallel training"""
    if hasattr(outputs, 'logits'):
        logits = outputs.logits
    else:
        logits = outputs
    device = logits.device

    batch_size = logits.shape[0]
    logits = logits.view(-1, logits.shape[-1])
    labels = labels.flatten().to(device)
    sploss_parallel_size = int(os.environ.get('CELOSS_PARALLEL_SIZE', '0'))
    if sploss_parallel_size > 0:
        from swift.trainers.sequence_parallel.utils import ChunkedCrossEntropyLoss
        loss = ChunkedCrossEntropyLoss.apply(logits, labels, sploss_parallel_size)
    else:
        loss_fct = CrossEntropyLoss(reduction='none')
        loss = loss_fct(logits, labels)
    if enable_dft_loss:
        with torch.no_grad():
            target_probs = torch.exp(-loss)
        loss *= target_probs
    from swift.trainers.sequence_parallel import sequence_parallel
    position_ids = sequence_parallel.real_position_ids
    if position_ids is not None:
        position_ids = sequence_parallel.pad(position_ids, padding_value=-1, position_ids=position_ids)
    from swift.trainers.sequence_parallel.utils import GatherLoss
    loss, labels = GatherLoss.apply(loss.reshape(batch_size, -1), labels.reshape(batch_size, -1), 1, position_ids)
    if position_ids is not None and position_ids.min() == -1:
        _pos_mask = position_ids >= 0
        loss = loss[_pos_mask].contiguous()

    return loss


def per_token_loss_func(outputs, labels, enable_dft_loss: bool = False, **kwargs):
    logits = outputs.logits
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()
    labels = torch.roll(labels, shifts=-1, dims=-1).view(-1)

    # Flatten the tokens
    logits = logits.view(-1, logits.shape[-1])
    # Enable model parallelism
    labels = labels.to(logits.device)
    loss = F.cross_entropy(logits, labels, ignore_index=-100, reduction='none')
    if enable_dft_loss:
        with torch.no_grad():
            target_probs = torch.exp(-loss)
        loss *= target_probs
    return loss


def extract_version(name: str) -> Optional[int]:
    if not name.startswith('v'):
        return None
    try:
        num = name[1:].split('-', 1)[0]
        return int(num)
    except ValueError:
        return None


def get_previous_version_from_path(current_path: str) -> Optional[str]:
    from pathlib import Path
    current = Path(current_path)
    parent = current.parent
    current_name = current.name

    candidates = [d for d in parent.iterdir() if d.is_dir()]
    valid = [(d.name, extract_version(d.name)) for d in candidates]
    valid = [(name, ver) for name, ver in valid if ver is not None]
    valid.sort(key=lambda x: x[1])
    names = [name for name, _ in valid]

    if current_name not in names:
        return None

    idx = names.index(current_name)
    if idx == 0:
        return None

    prev_name = names[idx - 1]
    return str(parent / prev_name)


def get_resume_dir(output_dir):
    return get_previous_version_from_path(output_dir)
