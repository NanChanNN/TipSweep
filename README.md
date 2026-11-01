<div align="center">

<!-- Replace the author href="#" placeholders and the Paper / Project Page
     destinations when public URLs are available. Update the arXiv badge and
     its link when a public identifier is available. -->

<h1>TipSweep</h1>

<p><strong>Precision-Aware Learning Rate Transfer<br>Across Block Count in LLM Post-Training</strong></p>

<p>
  <a href="#" title="Author homepage placeholder">Nan Chen</a> ·
  <a href="https://www.ams.jhu.edu/villar/" title="Soledad Villar's homepage">Soledad Villar</a> ·
  <a href="https://www.soufianehayou.com/" title="Author homepage placeholder">Soufiane Hayou</a><br>
Department of Applied Mathematics and Statistics, Johns Hopkins University</p>

<p><em>Tune a small trainable suffix. Transfer the learning rate to a larger set of blocks.</em></p>

[![Arxiv](https://img.shields.io/badge/arXiv-link%20pending-B31B1B?style=flat-square&logo=arxiv&logoColor=white)](#citation "Paper URL placeholder")
[![License](https://img.shields.io/badge/License-MIT-2563EB?style=flat-square)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)

<p>
  <a href="#citation" title="Paper URL placeholder. See citation details">Paper</a>  | 
  <a href="#setup">Quickstart</a>  | 
  <a href="#citation">Citation</a>
</p>

</div>

## Method

TipSweep searches for a learning rate by training only the last $k_0$ transformer blocks of a pretrained model. It then transfers the selected rate $\eta_{k_0}$ to a run with $k$ trainable blocks, starting from the same checkpoint and using the same update precision.

| Parameter-update precision | Learning rate transferred from $k_0$ to $k$ |
| :--- | :--- |
| FP32 accumulation, including FP32 master weights | $\eta_k \approx \eta_{k_0}\,(k_0/k)$ |
| Direct BF16 updates, without an FP32 master copy | $\eta_k \approx \eta_{k_0}\,\sqrt{k_0/k}$ |

All parameters within the selected blocks are trainable, including normalization weights. The remaining blocks, input embeddings, final normalization, and language-model head stay frozen. **Full-depth** means training all transformer blocks, with these other components still frozen. **Full-parameter** means training all model parameters. Full-parameter training can use the same learning rate as full-depth training in practice.

For a fixed offline dataset, the frozen prefix can be evaluated once and its features reused across proxy trials. The paper studies supervised fine-tuning (SFT), direct preference optimization (DPO), and an extension to reinforcement learning with verifiable rewards (RLVR). Reusable feature caching applies to the off-policy setting but not to the on-policy RLVR because the policy model generates new completions on-the-fly during training. The paper reports approximately **8× lower analytical FLOPs** for a 16-trial Llama-2-7B search, including cache construction and the final full-depth run. Its prototype demonstrates a **6.7B-model proxy search on a consumer GPU with about 16 GiB of memory**. The final full-depth run still uses high-end GPUs.

## Setup

Run commands from the repository root in a CUDA-enabled environment:

```bash
python -m pip install -r requirements.txt
```

[requirements.txt](requirements.txt) pins the training stack, including PyTorch, Transformers, TRL, Accelerate, DeepSpeed, and vLLM. Model checkpoints and datasets are loaded from Hugging Face. Please configure access to gated checkpoints before launching a run.

Two workflows use specific attention backends:

- **Cached prototype:** defaults to [FlashAttention 2](https://github.com/Dao-AILab/flash-attention). Install a build compatible with your PyTorch/CUDA environment, or pass `--attn-impl eager` to both stages.
- **RLVR:** uses colocated vLLM and the Hugging Face [Kernels](https://huggingface.co/docs/kernels/main/en/installation) backend `kernels-community/vllm-flash-attn3`, which requires compatible GPU hardware.

Memory and timing figures depend on the hardware, attention backend, and batch configuration.

## Experiments

### Entry points

| Workflow | Script | Models / data |
| :--- | :--- | :--- |
| SFT, FP32 / BF16 / FP16 | [run-llama-depth-sweep.py](run-llama-depth-sweep.py) | Llama · Tülu 3 |
| Code SFT, direct BF16 | [run-mistral-code-sweep-bf16.py](run-mistral-code-sweep-bf16.py) | Mistral / DeepSeek-Coder · Magicoder |
| Code SFT, FP32 accumulation | [run-mistral-code-sweep-fp32.py](run-mistral-code-sweep-fp32.py) | Mistral / DeepSeek-Coder · Magicoder |
| DPO, FP32 / BF16 | [run-olmo-dpo-sweep.py](run-olmo-dpo-sweep.py) | Llama / Qwen · UltraFeedback<br>OLMo · olmo official preference mixture |
| RLVR, FP32 / BF16 | [run-llama-rlvr-depth-sweep.py](run-llama-rlvr-depth-sweep.py) | Llama-3.1-8B · GSM8K |
| Feature-cache construction | [run-prototype-s1.py](run-prototype-s1.py) | DeepSeek-Coder-6.7B · Magicoder |
| Cached proxy training | [run-prototype-s2.py](run-prototype-s2.py) | DeepSeek-Coder-6.7B · cached features |
| SFT profiling | [run-llama-depth-profiling.py](run-llama-depth-profiling.py) | Llama · Tülu 3 |
| DPO profiling | [run-olmo-dpo-profiling.py](run-olmo-dpo-profiling.py) | Llama / Qwen · UltraFeedback |

Each sweep-script invocation runs **one learning-rate / block-count configuration**.

### Shared conventions

- **Learning rate:** `--learning-rate-exponent a` sets $\eta = 2^{-a}$. Half-integer values such as `11.5` are supported.
- **Trainable depth:** `--last-n k` selects the final $k$ blocks. Use the model's total block count for full-depth training: 16 for Llama-3.2-1B, or 32 for DeepSeek-Coder-6.7B.
- **Effective batch size:** number of processes × per-device batch size × gradient accumulation steps. For RLVR, this counts completions. Divide by `--num-generations` to obtain the number of problems.
- **Precision flags:** Llama SFT and DPO use `--data-type`, RLVR uses `--model-dtype`, and cached trials use `--precision`.

The Llama and direct-BF16 code SFT scripts also support `--first-n` and `--sample-n`, for example `--sample-n 4-S42`. The direct-BF16 code SFT script supports attention-only or MLP-only training through `--ft-mode attn` or `--ft-mode mlp`.

### Llama SFT on Tülu 3

Train the final block of Llama-3.2-1B with FP32 parameter accumulation, a constant learning rate, zero weight decay, and an effective batch size of 32:

```bash
python run-llama-depth-sweep.py \
  --base-model Llama-3.2-1B \
  --data-type fp32 \
  --last-n 1 \
  --learning-rate-exponent 14 \
  --batch-size 4 --grad-accum 8 \
  --weight-decay 0 \
  --warmup-ratio 0 --lr-scheduler-type constant \
  --train-set-size 10000 --test-set-size 1000
```

In this script, the two set-size arguments are multiplied by 32: the values above request **320,000 training** and **32,000 held-out** examples. Epoch checkpoints are saved under `checkpoints/`, with precision included in the run path.

For direct BF16 updates, use `--data-type bf16`. For direct FP16 updates with FP32 optimizer moments, use `--data-type fp16 --fp32-optim-states`. 

<details>
<summary><strong>Code SFT · DeepSeek-Coder on Magicoder</strong></summary>

These full-depth examples use the learning rates transferred from the paper's last-block proxy: $2^{-14}$ for BF16 and $2^{-17}$ for FP32. Both have an effective batch size of 32, cosine decay with 5% warmup, and zero weight decay. The paper runs full-depth training on H200 GPUs.

**Direct BF16, one GPU:**

```bash
python run-mistral-code-sweep-bf16.py \
  --base-model deepseek-coder-6.7b-base \
  --last-n 32 --learning-rate-exponent 14 \
  --batch-size 4 --grad-accum 8 \
  --filter-length --save-checkpoint
```

**FP32 master weights, two GPUs with DeepSpeed ZeRO-3:**

```bash
accelerate launch --use_deepspeed --num_processes 2 --mixed_precision bf16 \
  run-mistral-code-sweep-fp32.py \
  --base-model deepseek-coder-6.7b-base \
  --last-n 32 --learning-rate-exponent 17 \
  --batch-size 8 --grad-accum 2 \
  --filter-length --save-checkpoint
```

`--filter-length` applies the bundled model-specific exclusion indices in [auxiliary_data/](auxiliary_data/). The FP32 script uses [the supplied ZeRO-3 configuration](configs/ds_configs/stage3_no_offloading_accelerate.conf) by default. To train Mistral instead, use `--base-model Mistral-7B-v0.1` and choose a learning rate for that setting.

</details>

<details>
<summary><strong>DPO · Llama on UltraFeedback and OLMo on its preference mixture</strong></summary>

**Llama-3.2-1B-Instruct:** standard sigmoid DPO with $\beta=0.1$, constant learning rate, and effective batch size 8.

```bash
python run-olmo-dpo-sweep.py \
  --base-model Llama-3.2-1B-Instruct \
  --task ultrafeedback_binarized \
  --data-type fp32 \
  --last-n 1 --learning-rate-exponent 18.5 \
  --per-device-batch-size 2 --grad-accum 4 \
  --warmup-ratio 0 --lr-scheduler-type constant \
  --loss-type sigmoid --beta 0.1 \
  --save-checkpoint
```

**OLMo-2-1124-7B-SFT:** length-normalized sigmoid DPO with $\beta=5$, 24,000 training pairs, 2,400 validation pairs, and effective batch size 8. The rate below is an illustrative sweep candidate.

```bash
accelerate launch --use_deepspeed --num_processes 2 --mixed_precision bf16 \
  run-olmo-dpo-sweep.py \
  --base-model OLMo-2-1124-7B-SFT \
  --task olmo-2-1124-7b-preference-mix \
  --data-type fp32 \
  --last-n 1 --learning-rate-exponent 18 \
  --per-device-batch-size 2 --grad-accum 2 \
  --train-set-size 24000 --test-set-size 2400 \
  --warmup-ratio 0 --lr-scheduler-type constant \
  --loss-type sigmoid_norm --beta 5 \
  --save-checkpoint
```

The script precomputes reference log probabilities and uses the templates in [chat_templates/](chat_templates/). For direct BF16 DPO, use `--data-type bf16` and launch without DeepSpeed.

</details>

<details>
<summary><strong>RLVR · Llama-3.1-8B on GSM8K</strong></summary>

This two-GPU example uses GRPO with token-level DAPO loss, 100 optimizer steps, and the paper's batch size of **128 problems × 16 completions**. It uses FP32 master weights through DeepSpeed with an illustrative candidate learning rate.

```bash
accelerate launch --use_deepspeed --num_processes 2 --mixed_precision bf16 \
  run-llama-rlvr-depth-sweep.py \
  --base-model Llama-3.1-8B \
  --model-dtype fp32 \
  --last-n 1 --learning-rate-exponent 16 \
  --batch-size 32 --grad-accum 32 \
  --num-generations 16 --training-steps 100 \
  --save-checkpoint
```

The script uses [the GRPO ZeRO-3 configuration](configs/ds_configs/stage3_no_offloading_grpo.conf), colocated vLLM generation, and the reward functions in [utils/reward_utils.py](utils/reward_utils.py). For a direct-BF16 distributed run, use `--model-dtype bf16` and replace the launcher's `--use_deepspeed` with `--multi_gpu`.

</details>

## Cached proxy prototype

The prototype splits DeepSeek-Coder-6.7B training into two stages. With `--output-layer 31`, Stage 1 evaluates the first 31 blocks and Stage 2 trains the final block. Both stages must use the same split and cache root.

### 1. Build the frozen-prefix cache

```bash
python run-prototype-s1.py \
  --output-layer 31 \
  --batch-size 1 \
  --filter-length \
  --save-root checkpoints
```

Features are stored in BF16 at:

```text
checkpoints/deepseek-coder-6.7b-base/Magicoder-OSS-Instruct-75K/FT-O31/feature_cache/
```

`--filter-length` applies the bundled exclusion indices during cache construction. Rebuilding a cache removes the existing `batch_*.safetensors` files in that destination before extraction while leaving unrelated files intact.

### 2. Run a cached trial

Reuse the same cache for either update precision. Each invocation makes one pass over the cached dataset, and both examples have an effective batch size of 32.

```bash
# Direct BF16 updates
python run-prototype-s2.py \
  --output-layer 31 --precision bf16 \
  --batch-size 1 --grad-accum 32 \
  --learning-rate-exponent 11.5 \
  --cache-root checkpoints

# FP32 trainable weights and optimizer states, with BF16 autocast
python run-prototype-s2.py \
  --output-layer 31 --precision fp32 \
  --batch-size 2 --grad-accum 16 \
  --learning-rate-exponent 12 \
  --cache-root checkpoints
```

These are the proxy learning rates selected in the paper's case study. Their transfers to all 32 blocks are:

| Update precision | Last-block proxy LR | Full-depth LR | Target CLI exponent |
| :--- | :--- | :--- | :--- |
| Direct BF16 | $2^{-11.5}$ | $2^{-14}$ | `14` |
| FP32 accumulation | $2^{-12}$ | $2^{-17}$ | `17` |

Use the code SFT entry points above for the final full-depth run.

## Profiling

The profiling entry points run on a **single GPU** with a positive `--last-n`. They save records under `profiling_records/` as `*_record.pt` files.

| `--profile-type` | Measurements |
| :--- | :--- |
| `loss-space-update` | Per-block update responses, gradient inner products, and loss-space statistics |
| `rounding-scan` | BF16 update-rounding factors and fractions of nonzero updates across probe learning rates |

For example, profile the first 3,000 updates of the FP32 Llama SFT last-block run:

```bash
python run-llama-depth-profiling.py \
  --base-model Llama-3.2-1B --data-type fp32 \
  --last-n 1 --learning-rate-exponent 14 \
  --profile-type loss-space-update \
  --max-steps 3000 --logging-steps 1 \
  --batch-size 4 --grad-accum 8 \
  --weight-decay 0 \
  --warmup-ratio 0 --lr-scheduler-type constant
```

For rounding probes, select `--profile-type rounding-scan`. Use `--measure-every` to control the interval and `--scan-exponents` to control the probe learning-rate grid. DPO profiling uses [run-olmo-dpo-profiling.py](run-olmo-dpo-profiling.py), which supports Llama and Qwen on UltraFeedback despite its filename. Use each script's `--help` for its full argument list.

## Citation

If you use this work, please cite out paper:

```bibtex
@misc{chen_tipsweep,
  title  = {Precision-Aware Learning Rate Transfer Across Block Count in {LLM} Post-Training},
  author = {Chen, Nan and Villar, Soledad and Hayou, Soufiane},
  note   = {Preprint},
  url    = {https://github.com/NanChanNN/TipSweep}
}
```

## License

This repository is released under the [MIT License](LICENSE).
