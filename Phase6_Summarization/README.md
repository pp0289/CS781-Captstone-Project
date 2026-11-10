# Phase 6 — Abstractive Summarization

**Notebook:** `phase_6.ipynb`

## Task

| | |
|---|---|
| Input | Full text of an Indian judgment |
| Output | `[{"id": ..., "summary": "..."}, ...]` |
| Metric | BERTScore F1 (ROUGE-1/2/L are also reported) |

## Data (`Phase_6_Summ`)

* **Splits:** 7,030 training pairs, of which **300 are held out** as a local validation set (there is no public dev split). The test set has 100 documents.
* **Length:** documents have a median of 3,130 words (p90 7,856) and summaries 637 words (p90 1,496). The median compression is 4.6×.
* **Truncation:** 94.3 % of documents exceed the 1,280-token input budget, and 60.4 % of summaries exceed the 640-token target budget.

**Coverage analysis** (model-free). It measures the share of reference content tokens, excluding the 150 most frequent, that appear in the input window. Coverage grows from 0.19–0.25 at 384 tokens to 0.60–0.65 at 2,304 tokens. At 1,536 tokens:

| Input strategy | Coverage |
|---|---|
| head | 0.495 |
| tail | 0.552 |
| head + tail | 0.553 |
| sentence selection | 0.541 |

## Method

**Format.** A decoder-only (causal) formulation that reuses the Phase-1 LM head:

```
<s> [document: 1,280 tokens = first 60 % + last 40 %] "\n\nSummary:" [summary ≤ 640 tokens] </s>
```

**Loss.** Cross-entropy on **summary tokens only**.

**Decoding.** Greedy with a KV cache:
* 60–512 new tokens;
* repetition penalty 1.15 over the last 128 tokens;
* no repeated 4-grams;
* a trailing unfinished sentence is trimmed at the length cap;
* if fewer than 5 words are produced, the first 100 words of the document are used instead.

**Model selection.** Held-out cross-entropy on all 300 documents.
Generation metrics are computed on the first 150 held-out documents: ROUGE via `rouge-score`, and BERTScore via `bert-score` with its default `roberta-large`, used only as an evaluator.



### Techniques and learning rates

| Method | LR | Trainable |
|---|---|---|
| Full FT | 5e-5 | all |
| Partial FT (top 4 blocks) | 1e-4 | 34.2M |
| Adapters (64-d) | 5e-4 | adapters only |
| LoRA / LoRMA (r = 8) | 5e-4 | attention projections only |
| KD student: blocks {0, 5, 10, 15} (53.3M) | 5e-5 | all of the student |

The batch is 4 documents × accumulation 4 (16 effective), with AdamW, 6 % warm-up, cosine decay and fp16.

## Results (150 held-out documents)

| Run | Val CE | BERTScore | ROUGE-1 | ROUGE-L | Words | Train time |
|---|---|---|---|---|---|---|
| Lead-N baseline (N = 705 words) | – | 0.8286 | 0.5160 | 0.2675 | 705 | – |
| Extractive selector baseline | – | 0.8321 | 0.5409 | 0.2782 | – | – |
| Clean GQA-RoPE full FT, 2 epochs | 2.1793 | 0.8426 | 0.4491 | 0.2298 | 360 | 54.1 min |
| Clean GQA-RoPE full FT, 5 epochs (best CE at epoch 3) | 2.1495 | **0.8480** | 0.4570 | 0.2391 | 351 | 134.1 min |
| Clean GQA-RoPE adapters, 3 epochs | – | 0.8455 | 0.4362 | 0.2312 | – | – |
| Clean GQA-RoPE partial FT | 2.4134 | 0.8427 | 0.4517 | 0.2267 | 367 | 48.8 min |
| Clean GQA-RoPE KD student | 3.1470 | 0.8305 | 0.3904 | 0.2027 | 316 | 29.2 min |
| MLA-RoPE LoRA | 3.5825 | 0.8277 | 0.3973 | 0.2023 | 362 | 56.8 min |
| MLA-Sin LoRA | 4.0409 | 0.8191 | 0.3965 | 0.2085 | 362 | 53.1 min |
| GQA-Sin LoRMA | 4.1501 | 0.8094 | 0.3921 | 0.2076 | 358 | 47.4 min |



## Submissions

| Codabench file | Run | Local BERTScore | Codabench |
|---|---|---|---|
| `submission_B__clean_gqa_rope_moe__full.zip` | Clean GQA-RoPE, full FT | 0.8426 | **0.8421** |
| `submission_B__clean_gqa_rope_moe__adapter.zip` | Clean GQA-RoPE, adapters | 0.8455 | 0.8409 |
| `submission_B__clean_gqa_rope_moe__partial.zip` | Clean GQA-RoPE, partial FT (top 4) | 0.8427 | 0.8385 |
| `submission_D__clean_gqa_rope_moe__kd.zip` | Clean GQA-RoPE, KD student | 0.8305 | 0.8272 |
| `submission_B__mla_rope_moe__lora.zip` | MLA-RoPE, LoRA | 0.8277 | 0.8242 |
| `submission_B__mla_sin_moe__lora.zip` | MLA-Sin, LoRA | 0.8191 | 0.8168 |
| `submission_B__gqa_sin_moe__lorma.zip` | GQA-Sin, LoRMA | 0.8094 | 0.8044 |


## How to run

1. Attach `phase1-models`, add the `HF_TOKEN` secret, run the setup section, and uncomment and run the `!cp` cell. Checkpoints are read from `phase1-models/best_models/best_models/`.
2. In the config cell, set `PROFILE = "full"` (`fast`/`balanced` shrink the data and lengths for quick tests), `BASE_VARIANT = "clean_gqa_rope_moe"` and `STUDENT_KEEP`.
3. Run data loading, EDA, the coverage analysis, tokenisation, the PEFT modules, the summariser and the decoding/evaluation utilities. `bert-score` downloads `roberta-large` for local scoring.
4. Run the baseline cell (lead-N, extractive selector), then the experiment cells (`B__…` techniques, `D__…` KD). They are cached and resumable under `summ_runs/`.
5. Run the submission cell. It writes `summ_runs/sub_<tag>/prediction.json` and `/kaggle/working/submission_<tag>.zip` with all 100 test summaries.

## Outputs

* `/kaggle/working/summ_runs/`: checkpoints, generated validation summaries and the results table.
* `/kaggle/working/submission_<tag>.zip`, which contains `prediction.json`.
