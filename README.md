<div align="center">

<h1>TipSweep</h1>

<p><strong>Precision-Aware Learning Rate Transfer<br>Across Block Count in LLM Post-Training</strong></p>

<p>
  <a href="https://nanchannn.github.io/" title="Nan Chen's homepage">Nan Chen</a> ·
  <a href="https://www.ams.jhu.edu/villar/" title="Soledad Villar's homepage">Soledad Villar</a> ·
  <a href="https://www.soufianehayou.com/" title="Soufiane Hayou's homepage">Soufiane Hayou</a><br>
Department of Applied Mathematics and Statistics, Johns Hopkins University</p>

<p><em>Tune a small trainable suffix. Transfer the learning rate to a larger set of blocks.</em></p>

[![Arxiv](https://img.shields.io/badge/arXiv-2611.31166-B31B1B?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2611.31166)
[![License](https://img.shields.io/badge/License-MIT-2563EB?style=flat-square)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)

<p>
  <a href="https://arxiv.org/abs/2611.31166" title="arXiv:2611.31166">Paper</a>  | 
  <a href="#setup">Quickstart</a>  | 
  <a href="#citation">Citation</a>
</p>

</div>

## Method

TipSweep searches for a learning rate by training only the last $k_0$ transformer blocks of a pretrained model. It then transfers the selected rate $\eta_{k_0}$ to a run with $k$ trainable blocks, starting from the same checkpoint and using the same update precision. We show that our method allows excellent lr transfer even with $k_0=1$.

| Parameter-update precision | Learning rate transferred from $k_0$ to $k$ |
| :--- | :--- |
| FP32 accumulation, including FP32 master weights | $\eta_k \approx \eta_{k_0}\,(k_0/k)$ |
| Direct BF16 updates, without an FP32 master copy | $\eta_k \approx \eta_{k_0}\,\sqrt{k_0/k}$ |

A **full-depth** run trains all $L$ transformer blocks ($k = L$) and keeps the input embeddings, final normalization, and language-model head frozen. A **full-parameter** run also trains these components and can use the same learning rate as the full-depth run.

On a fixed dataset, as in SFT and DPO, the output of the frozen blocks is the same in every trial, so it is computed once and shared. This does not apply to on-policy RLVR. For a 16-trial search on Llama-2-7B, this cuts analytical FLOPs by about **8×** compared with running every trial at full depth, including the cache build and the final full-depth run.

Two [prototypes](#learning-rate-search-prototypes) implement the search for a 6.7B model. One fits on a **consumer GPU with about 16 GiB**. The other completes a **five-candidate search in about half the GPU hours of one full-depth run**.

## Setup

Run all commands from the repository root in a CUDA-enabled environment:

```bash
python -m pip install -r requirements.txt
```

Models and datasets are loaded from Hugging Face, so set up access to gated checkpoints first.

## Experiments

### Entry points

| Workflow | Script | Models / data |
| :--- | :--- | :--- |
| SFT, FP32 / BF16 / FP16 | [run-llama-depth-sweep.py](run-llama-depth-sweep.py) | Llama · Tülu 3 |
| Code SFT, direct BF16 | [run-mistral-code-sweep-bf16.py](run-mistral-code-sweep-bf16.py) | Mistral / DeepSeek-Coder · Magicoder |
| Code SFT, FP32 accumulation | [run-mistral-code-sweep-fp32.py](run-mistral-code-sweep-fp32.py) | Mistral / DeepSeek-Coder · Magicoder |
| DPO, FP32 / BF16 | [run-olmo-dpo-sweep.py](run-olmo-dpo-sweep.py) | Llama / Qwen · UltraFeedback<br>OLMo · OLMo preference mixture |
| RLVR, FP32 / BF16 | [run-llama-rlvr-depth-sweep.py](run-llama-rlvr-depth-sweep.py) | Llama-3.1-8B · GSM8K |
| Learning rate search prototypes | [prototypes/](prototypes/) | DeepSeek-Coder-6.7B · Magicoder |
| SFT profiling | [run-llama-depth-profiling.py](run-llama-depth-profiling.py) | Llama · Tülu 3 |
| DPO profiling | [run-olmo-dpo-profiling.py](run-olmo-dpo-profiling.py) | Llama / Qwen · UltraFeedback |

Each run of a sweep script trains one learning rate with one number of trainable blocks.

### Shared conventions

- **Learning rate:** `--learning-rate-exponent a` sets $\eta = 2^{-a}$. Half-integers such as `11.5` work.
- **Trainable blocks:** `--last-n k` trains the last $k$ blocks. Use 16 for full-depth Llama-3.2-1B and 32 for DeepSeek-Coder-6.7B. Some scripts also accept `--first-n k` for the first $k$ blocks, and `--sample-n 4-S42` for 4 random blocks chosen with seed 42.
- **Modules:** `--ft-mode attn` or `--ft-mode mlp` trains only the attention or MLP modules (direct-BF16 code SFT only).
- **Effective batch size:** processes × per-device batch size × gradient accumulation steps. For RLVR, this counts completions; divide by `--num-generations` to get the number of problems.
- **Precision:** `--data-type` for Llama SFT and DPO, `--model-dtype` for RLVR, and `--precision` for the prototypes.

### Llama SFT on Tülu 3

Train the last block of Llama-3.2-1B with FP32 updates:

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

Use `--data-type bf16` for direct BF16 updates, or `--data-type fp16 --fp32-optim-states` for direct FP16 updates with FP32 optimizer states for stability.

<details>
<summary><strong>Code SFT · DeepSeek-Coder on Magicoder</strong></summary>

Full-depth runs at the rates transferred from the paper's last-block search: $2^{-14}$ for BF16 and $2^{-17}$ for FP32.

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

For Mistral, use `--base-model Mistral-7B-v0.1` with a learning rate chosen for that model.

</details>

<details>
<summary><strong>DPO · Llama on UltraFeedback and OLMo on its preference mixture</strong></summary>

**Llama-3.2-1B-Instruct**, sigmoid DPO with $\beta=0.1$:

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

**OLMo-2-1124-7B-SFT**, length-normalized DPO with $\beta=5$, at an example learning rate from the sweep:

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

For direct BF16 updates, use `--data-type bf16` and launch without DeepSpeed.

</details>

<details>
<summary><strong>RLVR · Llama-3.1-8B on GSM8K</strong></summary>

GRPO with the token-level DAPO loss on two GPUs, using the paper's batch of 128 problems × 16 completions and an example learning rate:

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

For direct BF16 updates, use `--model-dtype bf16` and replace `--use_deepspeed` with `--multi_gpu`. The rewards are defined in [utils/reward_utils.py](utils/reward_utils.py).

</details>

## Learning rate search prototypes

[prototypes/](prototypes/) contains two prototypes of the search. Both train only the last block of DeepSeek-Coder-6.7B on Magicoder:

- **Two-stage** (`run-prototype-s1.py`, `run-prototype-s2.py`) caches the output of the frozen blocks on disk, then trains one candidate learning rate per run.
- **Cache-free** (`run-cache-free-prototype.py`) keeps that output in GPU memory and trains all candidates in a single pass.

Run them as modules from the repository root:

```bash
python -m prototypes.run-cache-free-prototype \
  --output-layer 31 --precision bf16 \
  --batch-size 16 --grad-accum 2 \
  --candidate-exponents 10.5 11 11.5 12 12.5 \
  --filter-length
```

The scripts log the smoothed training loss, and the cache-free prototype marks the best candidate at the end. See [prototypes/README.md](prototypes/README.md) for details.

## Profiling

The profiling scripts run on one GPU with a positive `--last-n` and save `*_record.pt` files under `profiling_records/`.

| `--profile-type` | Measurements |
| :--- | :--- |
| `loss-space-update` | Per-block update responses, gradient inner products, and loss-space statistics |
| `rounding-scan` | BF16 update-rounding factors and fractions of nonzero updates across probe learning rates |

Profile the first 3,000 updates of the FP32 Llama SFT run that trains the last block:

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

For rounding probes, use `--profile-type rounding-scan` with `--measure-every` (interval) and `--scan-exponents` (probe learning rates). [run-olmo-dpo-profiling.py](run-olmo-dpo-profiling.py) profiles DPO for Llama and Qwen on UltraFeedback. Run any script with `--help` for all options.

## Citation

If you find this work helpful, please cite our paper:

```bibtex
@misc{chen2026precisionaware,
  title         = {Precision-Aware Learning Rate Transfer Across Block Count in {LLM} Post-Training},
  author        = {Chen, Nan and Villar, Soledad and Hayou, Soufiane},
  year          = {2026},
  eprint        = {2611.31166},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2611.31166},
  url           = {https://arxiv.org/abs/2611.31166}
}
```

## License

This repository is released under the [MIT License](LICENSE).
