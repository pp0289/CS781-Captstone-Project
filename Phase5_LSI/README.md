# Phase 5 — Legal Statute Identification (LSI)

**Notebook:** `phase_5.ipynb`

## Task

| | |
|---|---|
| Input | Facts / text of a case |
| Output | `[{"id": ..., "pred": [statute_index, ...]}, ...]`, a set of statutes out of 100 |
| Metric | Macro-F1 over the 100 statutes (Micro-F1 is also reported) |

## Data (`Phase_5_LSI`)

| Split | Docs |
|---|---|
| Train | 42,750 |
| Dev | 10,181 |
| Test | 13,019 |

* **Labels:** all 100 statute IDs appear in training, and none has fewer than 20 training documents. The frequency distribution still has a long tail.
* **Labels per document:** 3.70 on average; 21.4 % of documents are single-label.
* **Length:** median 446 words; 1,401 tokens per document after clipping.

## Method

### 1. Preprocessing and chunking
The pipeline is the same as Phase 4:
* whitespace is collapsed and only the first and last 12,000 characters are kept;
* text is tokenised once and cached;
* it is split into 511-token chunks prefixed with `<s>`;
* the final setting is **head+tail, K = 8** (the first 4 and last 4 non-overlapping chunks).

Targets are 100-dimensional multi-hot vectors over the official statute list.

### 2. Encoder
* **Chunk encoder.** The bidirectional Phase-1 encoder (padding-only mask, gradient checkpointing per block) encodes each chunk.
* **Chunk vector.** Each chunk is mean-pooled into `E_k ∈ R^768`.

### 3. Label-wise attention head (LWAN)
A single pooled document vector has to serve all 100 statutes at once. Instead, each statute gets its own view of the document:

```
score_{ℓk} = (E_k · Q_ℓ) / √d                 (masked for padding chunks)
a_{ℓk}     = softmax_k(score_{ℓk})
d_ℓ        = dropout( Σ_k a_{ℓk} E_k )
z_ℓ        = ⟨d_ℓ, W_ℓ⟩ + b_ℓ
```

* **Parameters.** `Q_ℓ` and `W_ℓ` are per-label vectors (initialised N(0, 0.02)), and `b_ℓ` is a per-label bias. There is no shared MLP in this head.
* **Effect.** Each statute can attend to the chunk that mentions it: one label can rely on the facts in the head while another relies on the operative order in the tail.
* **Bias initialisation.** `b_ℓ` starts at the label's training log-odds, `log(f_ℓ / (1 − f_ℓ))`. Initial predictions therefore match label frequencies, and training does not start by pushing every logit down.

### 4. Training loss
* **Total loss.** `L = Σ_ℓ BCE(z_ℓ, y_ℓ)`, summed over the 100 labels and averaged over documents, plus `0.01 · L_aux`.


### 5. Threshold selection
1. **Grid.** Thresholds 0.05–0.95, step 0.05.
2. **Candidate strategies:**
   * **fixed** 0.5;
   * **global**: the single threshold with the best mean F1 over labels;
   * **per-label**: each label's own best threshold, falling back to the global one for labels with fewer than 5 dev positives.
3. **Choosing a strategy** (`choose`). For 3 random seeds, dev is split into two halves. Thresholds are tuned on one half and scored on the other, in both directions. The strategy with the best mean held-out Macro-F1 wins, and its thresholds are then refitted on the full dev set. **Per-label** won for both submitted models (cross-fitted 0.3940 versus 0.2956 at a fixed 0.5).
4. **At least one label.** A document whose probabilities are all below threshold still receives its top-scoring statute.
5. **Metric.** Macro-F1 is computed over labels that have at least one positive.

### 6. Fine-tuning methods
| Method | What is trained | Backbone LR |
|---|---|---|
| Full FT (clean GQA-RoPE) | everything | 3e-5 |
| LoRA / LoRMA, r = 8, α = 16 | attention projections + LWAN head | 1e-3 |
| KD student | Phase-1 blocks {0, 5, 10, 15} + LWAN head (53.5M), all trainable | 5e-5 |

**Knowledge distillation.**
* **Teacher cache.** The teacher's logits and document embeddings are precomputed once on the training set.
* **Student loss.** `L = 0.5 · BCE(z_s, y) + 0.5 · T² · BCE(z_s / T, σ(z_t / T)) + 0.5 · (1 − cos(d_s, d_t)) + 0.01 · L_aux`, with T = 2. The soft term is a sigmoid-based multi-label analogue of the KL used in Phase 4.

I have also implemented partial FT, gradual unfreezing, BitFit, a simplified AdaLoRA, adapters, prompt tuning and prefix tuning.

## Results (local dev)

| Run | macro-AP | Macro-F1 @0.5 | Macro-F1 (per-label thr.) | Cross-fitted | Micro-F1 | Time |
|---|---|---|---|---|---|---|
| Clean GQA-RoPE, full FT, 2 epochs | 0.3683 | 0.2956 | **0.4073** | 0.3940 | 0.503 | 250.8 min |
| Clean GQA-RoPE, KD student | 0.3677 | 0.2828 | 0.4066 | 0.3934 | 0.502 | 64.8 min |

* **Label count:** the full-FT model predicts 4.76 labels per test document, against 3.70 in training.
* **Teacher fit:** the teacher's training macro-AP is 0.542, against 0.368 on dev.

## Submissions

| Codabench file | Run | Local dev Macro-F1 | Codabench |
|---|---|---|---|
| `submission_B__clean_gqa_rope_moe__full.zip` | Clean GQA-RoPE, full FT, LWAN, per-label thresholds | 0.4073 | **0.3974** |
| `submission_D__clean_gqa_rope_moe__kd.zip` | Clean GQA-RoPE, KD student (53.5M) | 0.4066 | 0.3914 |
| `submission_B__mla_rope_moe__lora.zip` | MLA-RoPE, LoRA | – | 0.3230 |
| `submission_B__gqa_sin_moe__lora.zip` | GQA-Sin, LoRA | – | 0.3190 |
| `submission_B__mla_sin_moe__lorma_1.zip` | MLA-Sin, LoRMA | – | 0.3068 |



## How to run

1. Attach `phase1-models`, add the `HF_TOKEN` secret, run the setup section, and uncomment and run the `!cp` cell. Phases 5–7 read checkpoints from `phase1-models/best_models/best_models/abl_<variant>_best.pt`, including `abl_clean_gqa_rope_moe_best.pt`.
2. In the config cell, set:
   * `PROFILE = "full"`;
   * `BASE_VARIANT = "clean_gqa_rope_moe"`;
   * `STUDENT_KEEP = [0, 5, 10, 15]`;
   * `EFF_DOCS = 16`.
3. Run data loading and EDA, cached tokenisation, the PEFT modules, the classifier with the LWAN head, and the training/threshold utilities.
4. Run the experiment cells (`B__…` for techniques, `D__…` for KD). Runs are cached and resumable under `lsi_runs/`.
5. Run `write_sub(tag, p_test, thr)`. It writes `lsi_runs/sub_<tag>/prediction.json` and `submission_<tag>.zip`, after checking test IDs, unique in-range label indices and at least one label per document.

## Outputs

* `/kaggle/working/lsi_runs/`: checkpoints, cached probabilities and the results table.
* `submission_<tag>.zip`, copied to `/kaggle/working/` as well.
