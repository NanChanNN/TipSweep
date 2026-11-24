import argparse
import os

import torch
torch.backends.cuda.enable_cudnn_sdp(False)

from datasets import load_dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from utils.general_utils import configure_trainable_layers
from utils.data_utils import preprocess_gsm8k_simple
from utils.helper import naming_conversion
from utils.reward_utils import merged_gsm8k_reward_func


MAX_FORMATTED_PROMPT_TOKENS=512


global_rank = int(os.environ.get("RANK", "0"))
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))

master_process = global_rank == 0
distributed = world_size > 1


def main(args):
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/{args.base_model}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Dataset
    raw_data = load_dataset(f"openai/{args.task}", "main")
    train_set, test_set = raw_data['train'], raw_data['test']

    if args.test_set_size is not None:
        test_set = test_set.select(range(args.test_set_size))

    train_set = train_set.map(preprocess_gsm8k_simple, num_proc=16, remove_columns=train_set.column_names)
    test_set = test_set.map(preprocess_gsm8k_simple, num_proc=16, remove_columns=test_set.column_names)

    # Reward function
    def gsm8k_reward_func(completions, solution, log_metric=None, **kwargs):
        return merged_gsm8k_reward_func(
            completions=completions,
            solution=solution,
            log_metric=log_metric,
            accuracy_reward=args.accuracy_reward_point,
            format_reward=args.format_reward_point,
        )

    # grpo config
    grpo_args = GRPOConfig(
        output_dir=args.save_dir,
        model_init_kwargs=args.model_init_kwargs,
        cast_lm_head_to_fp32=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        optim="adamw_torch_fused",
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        max_grad_norm=args.max_grad_norm,
        max_steps=args.training_steps,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        lr_scheduler_kwargs=args.lr_scheduler_kwargs,
        save_strategy="steps" if args.save_checkpoint else 'no',
        save_steps=args.training_steps,
        bf16=True,
        tf32=True,
        deepspeed=args.deepspeed_config_file,
        gradient_checkpointing=args.enable_gradient_checkpointing,
        gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs,
        dataloader_drop_last=True,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        dataloader_prefetch_factor=2,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        disable_tqdm=False,
        seed=args.seed,
        # data preprocessing control
        num_generations=args.num_generations,
        num_generations_eval=args.num_generations_eval,
        max_completion_length=args.max_completion_length,
        ds3_gather_for_generation=True,
        pad_to_multiple_of=8,
        # generation control
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        # colocated vllm control
        use_vllm=True,
        vllm_mode='colocate',
        vllm_gpu_memory_utilization=0.40,
        vllm_max_model_length = (
            MAX_FORMATTED_PROMPT_TOKENS + args.max_completion_length
        ),
        vllm_tensor_parallel_size=1,
        vllm_enable_sleep_mode=True,
        # training control
        beta=args.kl_beta,
        num_iterations=1,
        importance_sampling_level='token',
        scale_rewards=args.scale_rewards,
        loss_type=args.loss_type,
        mask_truncated_completions=True,
        vllm_importance_sampling_correction=True,
        vllm_importance_sampling_mode='sequence_mask',
    )

    # configure trainer
    trainer = GRPOTrainer(
        model=f"meta-llama/{args.base_model}",
        args=grpo_args,
        processing_class=tokenizer,
        reward_funcs=[gsm8k_reward_func],
        train_dataset=train_set,
        eval_dataset=test_set,
    )
    
    # set up trainable layers
    configure_trainable_layers(model=trainer.model,
                               last_n=args.last_n)
    trainer.model.config.use_cache = False
    
    # training
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Basic arguments
    parser.add_argument('--base-model', type=str,
                        choices=["Llama-3.1-8B"],
                        default="Llama-3.1-8B")
    parser.add_argument('--model-dtype', type=str,
                        choices=['fp32', 'bf16'],
                        default='fp32')
    parser.add_argument('--task', type=str,
                        choices=["gsm8k"],
                        default="gsm8k")

    # full finetuning relevant
    parser.add_argument('--last-n', type=int, required=True)
    parser.add_argument('--learning-rate-exponent', type=float, required=True)

    # training relevant
    parser.add_argument('--training-steps', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--grad-accum', type=int, default=32)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--adam-beta1', type=float, default=0.9)
    parser.add_argument('--adam-beta2', type=float, default=0.999)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--warmup-steps', type=int, default=10)
    parser.add_argument('--lr-scheduler-type', type=str, default='constant_with_warmup')
    parser.add_argument('--deepspeed-config-file', 
                        type=str,
                        default='configs/ds_configs/stage3_no_offloading_grpo.conf')
    
    # GRPO-specific
    parser.add_argument('--accuracy-reward-point', type=float, default=1.0)
    parser.add_argument('--format-reward-point', type=float, default=0.10)
    parser.add_argument("--num-generations", type=int, default=16)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--kl-beta", type=float, default=0.0)
    parser.add_argument("--scale-rewards", type=str, default="batch")
    parser.add_argument("--loss-type", type=str, default="dapo")
    parser.add_argument('--enable-gradient-checkpointing', action='store_true')

    # evaluation
    parser.add_argument("--num-generations-eval", type=int, default=4)
    parser.add_argument('--eval-batch-size', type=int, default=32)
    parser.add_argument('--test-set-size', type=int, default=None)
    parser.add_argument('--eval-steps', type=int, default=10)

    # saving
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--logging-steps', type=int, default=1)
    parser.add_argument('--save-root', type=str, default='checkpoints')
    parser.add_argument('--save-checkpoint', action='store_true')

    args = parser.parse_args()

    args.learning_rate = 2 ** (-args.learning_rate_exponent)

    args.suffix = ''

    if args.lr_scheduler_type == 'cosine_with_min_lr':
        args.lr_scheduler_kwargs = {'min_lr_rate': 0.1}
    else:
        args.lr_scheduler_kwargs = {}

    if args.enable_gradient_checkpointing:
        args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    else:
        args.gradient_checkpointing_kwargs = {}
    
    if args.model_dtype == 'bf16':
        args.deepspeed_config_file = None
        
    args.model_init_kwargs = {'dtype': 'bfloat16',
                              'attn_implementation': "kernels-community/vllm-flash-attn3"}

    project_suffix = '' if args.model_dtype == 'fp32' else f"-{args.model_dtype}"

    save_dir = os.path.join(args.save_root, f"{args.base_model}{project_suffix}", args.task, f'FT-L{args.last_n}{args.suffix}',
                            f'LR-{naming_conversion(args.learning_rate_exponent)}')

    if master_process:
        os.makedirs(save_dir, exist_ok=True)

    args.save_dir = save_dir

    main(args)
