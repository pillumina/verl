set -x

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=2,3,4,5

# WORKSPACE_HOME and DATA_HOME support custom path configuration.
WORKSPACE_HOME=/data/h00513115
DATA_HOME=/data/h00513115

pr_enable=False
over_sampling_batch_size=64

sp_size=4
num_gpu=4
tp_size=2
train_prompt_bsz=32
train_prompt_mini_bsz=8

rollout_batch_size=32
n_samples_per_prompt=8

max_prompt_length=1024
max_response_length=16384


rollout_max_response_length=16384

CKPTS_DIR=$WORKSPACE_HOME/logs/ckpt/qwen3_8b
model_path=$DATA_HOME/models/qwen3-8b
train_data=$DATA_HOME/datasets/dapo-math-17k/data/dapo-math-17k.parquet
valid_data=$DATA_HOME/datasets/aime-2024/data/train.parquet

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$train_data \
    data.val_files=$valid_data \
    data.train_batch_size=$train_prompt_bsz \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$train_prompt_mini_bsz \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tp_size \
    actor_rollout_ref.rollout.name=sglang \
    +actor_rollout_ref.rollout.enable_partial_rollout=$pr_enable \
    +actor_rollout_ref.rollout.over_sampling_batch_size=$over_sampling_batch_size \
    actor_rollout_ref.rollout.response_length=$rollout_max_response_length \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=$n_samples_per_prompt \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.nccl_timeout=1800 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.val_before_train=False \
    trainer.project_name='verl_grpo_example' \
    trainer.experiment_name='qwen3_8b_partial_rollout' \
    trainer.n_gpus_per_node=$num_gpu \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.total_epochs=50 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} $@