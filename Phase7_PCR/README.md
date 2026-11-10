# Phase 7 — Prior Case Retrieval (PCR)

**Notebook:** `phase_7.ipynb`

## Task

| | |
|---|---|
| Input | A query case and a split-specific candidate pool |
| Output | `{query_id: [7 candidate ids]}`, with 7 unique in-pool IDs per query and never the query itself |
| Metric | Micro-F1@7 |

## Data (`Phase_7_PCR`)

| Split | Queries | Candidate pool | Relevant per query (mean / median / max) |
|---|---|---|---|
| Train | 827 | 4,320 | 6.42 / 4 / 52 |
| Dev | 118 | 1,023 | 8.25 / 5 / 92 |
| Test | 237 | 1,727 | – |

* **Ceiling:** because some queries have many relevant cases, even a perfect top-7 ranker scores at most **0.618** Micro-F1@7 on dev.
* **Length:** documents average 3,211 tokens after keeping the first and last 10k characters.

## Method

### Text handling

* `<NAME>` / `<ORG>` / `<CITATION>` placeholders are removed.
* Queries and candidates share the same tokenizer, truncation and encoder.

### BM25 baseline (own implementation)

* Grid search on dev selects k1 = 1.8 and b = 0.9.
* Dev scores: F1@7 0.2212, R@100 0.4995, MRR 0.6332.

### Dense bi-encoder

* **Encoder:** the shared-weight Phase-1 encoder, run bidirectionally.
* **Input:** K = 4 head+tail chunks of 511 tokens each.
* **Pooling:** mean over tokens, then over chunks, then L2 normalisation.
* **Similarity:** cosine.

### Training

* **Loss:** multi-positive InfoNCE, τ = 0.05.
* **Batch:** 8 queries, each with 1 sampled positive and 1 hard negative from BM25 ranks 0–50, plus in-batch negatives. Other gold candidates in the batch are masked, not treated as negatives.
* **Chunk dropout:** 2 of the 4 chunks are kept at random during training.
* **Schedule:** 4 epochs; LR 3e-5 for full FT and 5e-4 for LoRA, LoRMA and QLoRA.
* **Model selection:** dev-subset Micro-F1@7 (118 queries against 920 candidates: all gold plus random fillers).

### Hybrid scoring

```
score = (1 − w) · z(s_dense) + w · z(s_BM25)
```

`z(·)` is a per-query z-score, and `w ∈ {0, 0.1, …, 1}` is tuned on the full dev pool.

### Techniques

* **Full FT**
* **QLoRA:** every frozen linear layer is stored as packed 4-bit NormalFloat with block-64 absmax, plus LoRA r = 8 on attention. Effective size 56.0M, 0.64M trainable.
* **LoRA** and **LoRMA**
* **Prompt tuning:** 20 virtual tokens, 0.02M trainable.
* **KD student:** blocks {0, 5, 10, 15}.

MaxSim (chunk-level late interaction), listwise and triplet losses, and dense were implemented. All submissions use mean pooling with cosine similarity.

## Results (full dev pool)

| Run | Dense F1@7 | Hybrid F1@7 (best w) |
|---|---|---|
| BM25 only | – | 0.2212 |
| Zero-shot Phase-1 encoder (no training) | 0.2468 (R@100 0.637, MRR 0.586) | – |
| Clean GQA-RoPE full FT | 0.2935 (P@7 0.318, R@7 0.270, R@100 0.694) | **0.3180** (w = 0.2) |
| Clean GQA-RoPE QLoRA | 0.2946 | 0.3113 (w = 0.1) |
| GQA-Sin LoRA | 0.2513 | 0.2913 (w = 0.3) |
| MLA-RoPE LoRMA | 0.2501 | 0.3024 (w = 0.3) |
| MLA-Sin prompt tuning | 0.2023 | 0.2802 |

## Submissions

| Codabench file | Run | Dev F1@7 | Codabench |
|---|---|---|---|
| `submission_B__clean_gqa_rope_moe__full_hybrid.zip` | Clean GQA-RoPE full FT + BM25 hybrid | 0.3180 | **0.3142** |
| `submission_B__clean_gqa_rope_moe__qlora_hybrid.zip` | Clean GQA-RoPE QLoRA + hybrid | 0.3113 | 0.3012 |
| `submission_B__gqa_sin_moe__lora_hybrid.zip` | GQA-Sin LoRA + hybrid | 0.2913 | 0.2965 |
| `submission_D__clean_gqa_rope_moe__kd.zip` | Clean GQA-RoPE KD student | – | 0.2859 |
| `submission_B__mla_rope_moe__lorma_hybrid.zip` | MLA-RoPE LoRMA + hybrid | 0.3024 | 0.2823 |
| `submission_B__clean_gqa_rope_moe__full.zip` | Clean GQA-RoPE full FT, dense only | 0.2935 | 0.2735 |
| `submission_B__mla_sin_moe__prompt_hybrid.zip` | MLA-Sin prompt tuning + hybrid | 0.2802 | 0.2611 |

## How to run

1. Attach `phase1-models`, add the `HF_TOKEN` secret, run the setup section, and uncomment and run the `!cp` cell. Checkpoints are read from `phase1-models/best_models/best_models/`.
2. In the config cell, set `PROFILE = "full"`, `BASE_VARIANT` and `TOPK = 7`.
3. Run data loading and EDA, tokenisation, BM25 (the k1/b grid search) and the zero-shot encoder check.
4. Run the bi-encoder, PEFT and training cells, then the experiment cells. `B__…` covers the techniques; to run them on another backbone, set `BASE_VARIANT = "gqa_sin_moe"` / `"mla_rope_moe"` / `"mla_sin_moe"` in the corresponding cells. `D__…` covers KD. Runs are cached and resumable under `pcr_runs/`.
5. Run the submission cells. `write_sub(tag, S_test)` writes `pcr_runs/sub_<tag>/prediction.json` and `/kaggle/working/submission_<tag>.zip` for both the dense and the hybrid scores, after checking 7 unique in-pool IDs per test query. The best run by dev F1@7 is also copied to `/kaggle/working/submission.zip`.

## Outputs

* `/kaggle/working/pcr_runs/`: checkpoints, cached embeddings and score matrices.
* `/kaggle/working/submission_<tag>.zip` and `submission_<tag>_hybrid.zip`, each containing `prediction.json`.
