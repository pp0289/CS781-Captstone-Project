# Phase 1 — Legal Language-Model Pretraining

**Notebook:** `phase_1.ipynb`

## Task

| | |
|---|---|
| Input | A 512-word prompt from an Indian legal document |
| Output | A 128-word continuation |
| Metric | BERTScore F1 against the hidden reference continuation |
| Rules | 50M–300M parameters, random initialisation, official data only |

## Data

Training data comes from four official files: `Phase_1_Pretraining/Train/train_{acts,supreme_court,high_courts,tribunals}.json`.

| Source | Raw size | Share of sampled tokens |
|---|---|---|
| Acts | 0.73 GB | 25 % |
| Supreme Court | 1.32 GB | 25 % |
| High Courts | 10.7 GB | 30 % (capped; ~8× the SC file) |
| Tribunals | 6.48 GB | 20 % |

* **Sampling:** 200M training tokens and 5M validation tokens. Each document goes to validation with probability 0.05 until that source's quota is full.
* **Test set:** 1,000 prompts of 541–952 tokens (mean 613).

## Tokenizer and cleaning

* **Tokenizer:** byte-level BPE, retrained from our Assignment-1 implementation to **30,000** entries (4 reserved special tokens).
  * Trained on up to 100k cleaned documents per source, each at least 200 characters.
  * A GPT-2-style pre-tokenisation regex keeps leading spaces, so decoding is lossless.
* **Cleaner** (`clean_court_text`):
  * normalises Unicode quotes and dashes;
  * repairs mojibake;
  * removes IndianKanoon URLs, signatures and page numbers;
  * flattens tables and unwraps hard line breaks.

  A safety check compares dates, case numbers and court names before and after cleaning; if any of them changed, it falls back to conservative cleaning.
* **Training stream:** the 200M training tokens are encoded from the raw `text` field. Test prompts are cleaned before encoding.

## Architecture

The notebook implements four variants of a 16-layer, 768-wide decoder-only Transformer (section 9 of the notebook):

| | GQA-RoPE | MLA-RoPE | GQA-Sin | MLA-Sin |
|---|---|---|---|---|
| Attention | GQA, 2 KV heads × 96 | MLA: q-rank 384, KV latent 192, 80+16 q/k dims (decoupled RoPE), v 96 | GQA | MLA |
| Position | RoPE (θ = 10⁴) | RoPE | sinusoidal (additive) | sinusoidal |
| Total params | 152.1M (86.1M active) | 154.3M | 152.1M | 153.3M |

What all four variants share:

* **Normalisation and activation:** pre-norm RMSNorm, GELU, no biases, dropout 0.
* **Embeddings:** tied input/output embeddings.
* **Initialisation:** N(0, 0.02), with residual projections scaled to 0.02/√(2L).
* **FFN:** layers 0–1 use a dense FFN (2048). Layers 2–15 use an MoE with 8 routed experts (top-2, renormalised gates) plus 1 shared expert, each 512 wide.
* **Loss:** `L = CE + 0.01 · Σ_layers E · Σ_i f_i P_i` (Switch-style balance loss).

Checks built into the code:

* the parameter count is asserted to lie between 50M and 300M;
* a unit test checks KV-cache consistency (< 1e-7) and the backward pass for all 8 attention × position × FFN combinations.

## Training

| Setting | Value |
|---|---|
| Objective | next-token cross-entropy on random 1,024-token windows of the mixed stream |
| Tokens per step | 32,768 (micro-batch 4 × accumulation 8 × 1,024) |
| Optimiser | AdamW, β = (0.9, 0.95), weight decay 0.1 on matrices only |
| LR schedule | linear warm-up, cosine 6e-4 → 6e-5 |
| Clipping / precision | 1.0 / fp16 + GradScaler (T4) |
| Ablation budget | 30M tokens (915 steps) per variant, identical seed and schedule |
| Final model | GQA-RoPE continued to **100M tokens** (3,051 steps); the recomputed schedule restarts the LR at 5.35e-4 |
| Model selection | validation loss on positions ≥ 576 (mimics continuing after a ~512-word prompt), 40 random batches |


| Variant | Val. loss (pos ≥ 576) | Perplexity |
|---|---|---|
| GQA-RoPE | **2.568** | 13.0 |
| MLA-RoPE | 2.697 | 14.8 |
| GQA-Sin | 2.740 | 15.5 |
| MLA-Sin | 3.083 | 21.8 |
| GQA-RoPE, 100M tokens | **1.990** | 7.3 |

An earlier first attempt trained ~75M-parameter variants with 256-token blocks for 2,000 steps (peak LR 3e-4) before scaling up.

## Decoding (submission)

| Setting | Value |
|---|---|
| Input | full cleaned prompt, left-padded with an attention mask (checked against the causal path: max diff 1.05e-5) |
| Sampling | temperature 0.6, top-k 50, top-p 0.9 |
| Repetition control | repetition penalty 1.1 over the last 256 tokens; no repeated 6-grams |
| Banned tokens | special, whitespace-only and non-ASCII tokens |
| Length | stop once more than 128 words exist, then cut to exactly 128 words; short outputs are regenerated at a higher temperature (one retry was needed) |

The `TEMPERATURE`/`TOP_P` values in the config cell are not used. The generation cell sets its own values.

## Submissions

| Model | Pretraining tokens | Codabench BERTScore F1 |
|---|---|---|
| GQA-RoPE-MoE | 100M | **0.8325** |
| MLA-RoPE-MoE | 30M | 0.8281 |
| GQA-Sin-MoE | 30M | 0.8255 |
| MLA-Sin-MoE | 30M | 0.8247 |

## How to run

1. Add the `HF_TOKEN` Kaggle secret and select a GPU (T4).
2. Run sections 1–8 to set up, download data, train the tokenizer and build the token blocks. The tokens are cached in `/kaggle/working/cs781_phase1`.
3. Run section 9 to define the model and run the sanity test.
4. Run section 11 to train. It runs the 30M-token ablation for the four variants (`ABLATION_TOKENS`), then the GQA-RoPE continuation, and resumes automatically from `checkpoints/*_last.pt`.
5. Run sections 20–22 to generate predictions and write `prediction.json` and `submission.zip`.
6. Run the final cell to create `best_models.zip` (all `*_best.pt`). Upload it as the `phase1-models` Kaggle dataset used by Phases 2–7.

## Outputs

* `checkpoints/abl_{gqa,mla}_{rope,sin}_moe_best.pt`: best checkpoint per variant.
* `cs781_phase1/prediction.json`: `{"<id>": {"text": "<128-word continuation>"}, ...}` for all 1,000 prompts.
* `cs781_phase1/submission.zip`
* `training_validation_loss.png`
