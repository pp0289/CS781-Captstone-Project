# Phase 4 — Court Judgment Prediction (CJPE)

**Notebook:** `phase_4.ipynb`

## Task

| | |
|---|---|
| Input | Full text of a case |
| Output | `[{"id": ..., "pred": 0 or 1}, ...]` |
| Metric | Macro-F1 (Accuracy, Precision and Recall are also reported) |

## Data (`Phase_4_CJPE`)

| Split | Docs | Class balance |
|---|---|---|
| Train | 37,387 | 59.0 % class 0 / 41.0 % class 1 |
| Dev | 1,988 | 50 / 50 |
| Test | 1,517 | – |

The median document is 2,662 words.
After head+tail clipping, a document averages 2,741 tokens, about 5 chunks of 512 tokens.
## Method

### 1. Preprocessing and chunking
* **Clipping.** Whitespace is collapsed, and only the first and last 12,000 characters of each case are kept; very long middles are dropped. Each document is tokenised once and cached.
* **Chunks.** The token stream is split into 511-token chunks, and each chunk is prefixed with `<s>` (512 tokens in total).
* **Choosing K chunks** (`get_chunks`):

  | Strategy | Chunks taken |
  |---|---|
  | `head` | the first K forward chunks |
  | `tail` | the last K chunks, counted backwards from the end |
  | `headtail` | the first `K − ⌊K/2⌋` forward chunks plus the last `⌊K/2⌋` tail chunks that do not overlap them |
  | `uniform` | K evenly spaced forward chunks |

  A document shorter than one chunk is kept whole.
* **Batches.** A batch is a tensor `[B, K, 512]` plus a chunk mask. Only real chunks are flattened and encoded, so short documents cost less.

### 2. Hierarchical classifier
1. **Chunk encoder.** The Phase-1 embeddings, 16 blocks and final norm run with a padding-only mask (bidirectional). **Gradient checkpointing** is applied per block during training to fit 8 × 512 tokens per document on a T4.
2. **Chunk vector.** Masked mean over each chunk's tokens.
3. **Document pooling.** Five different pooling ways were used

   | Option | How it pools chunk vectors |
   |---|---|
   | `attn` | `Linear(768 → 256) → tanh → Linear(256 → 1)` scores, softmax over chunks |
   | `mean` | average |
   | `max` | element-wise max |
   | `last` | last real chunk |
   | `hier` | learned chunk-position embeddings, a 1-layer Transformer encoder over chunks (8 heads, FFN 1024), then attention pooling |

4. **Head.** MLP `768 → 384 → GELU → dropout 0.1 → 2`. A linear head is also implemented.

**Final configuration:** `headtail`, **K = 8** (≈ 4,088 content tokens), **mean** pooling, MLP head.

### 3. Training loss
* **Class weights.** Class-weighted cross-entropy with `w_c = n / (2 · n_c)`, computed on the training labels (59 % / 41 %).
* **Total loss.** `L = CE_weighted + 0.01 · L_aux` (MoE balance loss).

### 4. Model selection and threshold
* **Best epoch.** After each epoch the dev set is scored; the epoch with the best **dev AUROC** is kept. Dev loss, F1@0.5 and tuned F1 are also logged.
* **Threshold search.** Grid search over thresholds 0.05–0.95 (step 0.01) for the best Macro-F1.
* **Overfitting guard.** Dev is split once at random into two halves. The threshold tuned on each half is scored on the other half, and the two scores are averaged (cross-fitted F1). The tuned threshold is used only if cross-fitted F1 beats F1@0.5 by more than 0.005; otherwise 0.5 is kept. The best model used t = 0.73.

### 5. Fine-tuning methods
| Method | What is trained | Backbone LR |
|---|---|---|
| Full FT | everything | 3e-5 |
| Gradual unfreezing | head only in epoch 1 (backbone run under `no_grad`), then the top 4 → 8 → 16 blocks + final norm in later epochs | 1e-4 |
| LoRA / LoRMA, r = 8, α = 16 | attention projections + pooling/head | 1e-3 |
| KD student | Phase-1 blocks {0, 5, 10, 15} + pooling/head (53.6M), all trainable | 5e-5 |

**Knowledge distillation.**
* **Teacher cache.** The teacher (GQA-RoPE full FT) is run **once** over the whole training set, and its logits and document embeddings are cached (38.3 min). The student therefore never runs the 152M teacher during training.
* **Student loss.** `L = 0.5 · CE + 0.5 · T² · KL(p_teacher^T ‖ p_student^T) + 0.5 · (1 − cos(d_student, d_teacher)) + 0.01 · L_aux`, with T = 2. Here `d` is the pooled document vector.

The notebook also implements partial FT, BitFit (zero-initialised trainable biases), a simplified AdaLoRA (SVD-form update with importance pruning and an orthogonality penalty) and 64-d adapters.

## Results

Best model: GQA-RoPE full fine-tuning, 3 epochs, 485.5 min, 7.6 GB peak memory.


* **Macro-F1:** 0.7545 at threshold 0.5. The tuned threshold of **0.73** gives **0.7729** (P 0.773, R 0.773).
* **Codabench:** **0.7706**.

KD student: dev Macro-F1 0.7409, Codabench 0.7406, using 35 % of the parameters, 40.8 min of training and 3.2 GB peak memory.

## Submissions

| Codabench file | Run | Dev Macro-F1 (tuned) | Codabench |
|---|---|---|---|
| `submission_gqa_rope_moe_headtail_full.zip` | GQA-RoPE full FT, head+tail K = 8 | 0.7729 | **0.7706** |
| `submission_D__gqa_rope_moe__kd.zip` | GQA-RoPE KD student (53.6M) | 0.7409 | 0.7406 |
| `submission_B__gqa_sin_moe__lorma.zip` | GQA-Sin LoRMA | 0.6305 | 0.6178 |
| `submission_B__mla_sin_moe__lorma.zip` | MLA-Sin LoRMA | 0.6197 | 0.5932 |
| `submission_B__mla_rope_moe__gradual.zip` | MLA-RoPE gradual unfreezing (1 epoch) | 0.5924 | 0.5536 |




## How to run

1. Attach `phase1-models`, add the `HF_TOKEN` secret, run the setup section, and uncomment and run the `!cp` cell. Checkpoints are read from `phase1-models/best_models/abl_<variant>_best.pt`.
2. In **Cell 1** (imports and config), set:
   * `PROFILE = "full"`;
   * `BASE_VARIANT` (the variant used for the technique study and KD);
   * `STUDENT_KEEP`.
3. Run **Cells 2–7** in order:
   1. data and EDA;
   2. cached tokenisation and chunking;
   3. PEFT modules;
   4. hierarchical classifier;
   5. training and metrics;
   6. the cached, resumable experiment runner.
4. Run the **Study A** and **Study B** cells. Each experiment is cached under `cjpe_runs/`, so re-running skips finished runs and resumes interrupted ones.
5. Run the submission cells. They write `prediction.json` and `/kaggle/working/submission_<tag>.zip`, and check that the IDs match the test set exactly and that every prediction is 0 or 1.

## Outputs

* `/kaggle/working/cjpe_runs/`: checkpoints, cached logits and the results table.
* `/kaggle/working/submission_<tag>.zip`, which contains `prediction.json`.
