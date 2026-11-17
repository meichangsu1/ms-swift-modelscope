#!/usr/bin/env bash
# Async Full GRPO: FSDP2 on two NPUs, one TP2 rollout replica on two other NPUs.
set -euo pipefail
BENCHMARK_FRAMEWORK=swift
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
export USE_SWIFT_V5=1
cd "$SWIFT_DIR"

run_benchmark "$PYTHON_BIN" -m swift.dev.cli.rl \
    --rlhf_type grpo \
    --model "$MODEL_PATH" \
    --dataset "$DATA_DIR/swift_train.jsonl" \
    --dataset_shuffle false \
    --train_dataloader_shuffle false \
    --split_dataset_ratio 0 \
    --tuner full \
    --mode ray \
    --backend hf \
    --fsdp fsdp2 \
    --parallel_spec fsdp2 \
    --nproc_per_node 2 \
    --torch_dtype bfloat16 \
    --attn_impl sdpa \
    --fsdp_config '{"mixed_precision":{"param_dtype":"bf16","reduce_dtype":"fp32","output_dtype":"bf16"}}' \
    --gradient_checkpointing true \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --generation_batch_size 32 \
    --num_generations 4 \
    --num_iterations 1 \
    --dynamic_batching false \
    --dp_balance false \
    --padding_free false \
    --loss_type grpo \
    --scale_rewards group \
    --epsilon 0.2 \
    --epsilon_high 0.2 \
    --delta 3 \
    --beta 0 \
    --kl_in_reward false \
    --orm accuracy \
    --dynamic_sample false \
    --overlong_filter false \
    --rollout_importance_sampling_mode token_truncate \
    --rollout_importance_sampling_threshold 2.0 \
    --log_rollout_offpolicy_metrics false \
    --log_completions false \
    --enable_thinking false \
    --max_completion_length 2048 \
    --max_length 2560 \
    --temperature 1 \
    --top_p 1 \
    --top_k -1 \
    --sampler_mode disaggregated \
    --rollout_sampler vllm \
    --async_mode fully_async \
    --max_staleness "$MAX_STALENESS" \
    --weight_sync_backend full \
    --parameter_sync_step 1 \
    --vllm_tensor_parallel_size 2 \
    --vllm_data_parallel_size 1 \
    --vllm_gpu_memory_utilization "$ROLLOUT_MEMORY_FRACTION" \
    --vllm_enforce_eager true \
    --vllm_enable_prefix_caching true \
    --vllm_max_model_len 2560 \
    --vllm_max_num_seqs 32 \
    --vllm_engine_kwargs "{\"seed\":$SEED,\"generation_config\":\"vllm\",\"max_num_batched_tokens\":8192,\"enable_chunked_prefill\":true,\"additional_config\":{\"rl_config\":{\"enabled\":true}}}" \
    --sleep_level 0 \
    --offload_model false \
    --offload_optimizer false \
    --optim adamw_torch \
    --learning_rate 1e-6 \
    --lr_scheduler constant \
    --warmup_steps 0 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --adam_epsilon 1e-8 \
    --weight_decay 0 \
    --max_grad_norm 1 \
    --num_train_epochs 1 \
    --max_steps "$MAX_STEPS" \
    --seed "$SEED" \
    --save_strategy steps \
    --save_steps "$((MAX_STEPS + 1))" \
    --eval_strategy no \
    --logging_steps 1 \
    --report_to tensorboard \
    --logging_dir "$RUN_DIR/tensorboard" \
    --output_dir "$RUN_DIR"
