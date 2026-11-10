# Phase 3 — Rhetorical Role Prediction

**Notebook:** `phase_3.ipynb`

The notebook first repeats the Phase-1 setup and the Phase-2 code (sections 1–22 and "Phase 2 - L-NER").
The Phase-3 code starts at the **"Phase 3"** heading.

## Task

| | |
|---|---|
| Input | Pre-segmented sentences of Corporate-Law (CL) and IT judgments |
| Output | `{doc_id: [label_1, ..., label_n]}`, exactly one role per given sentence (CL and IT test documents combined in one file) |
| Labels (13) | Fact, Issue, ArgumentPetitioner, ArgumentRespondent, Statute, Dissent, PrecedentReliedUpon, PrecedentNotReliedUpon, PrecedentOverruled, RulingByLowerCourt, RatioOfTheDecision, RulingByPresentCourt, None |
| Metric | Macro-F1 |

## Data

| Domain | Docs (train / dev / test) | Sentences (train / dev / test) |
|---|---|---|
| CL | 40 / 5 / 5 | 10,593 / 1,396 / 1,339 |
| IT | 40 / 5 / 5 | 6,252 / 746 / 858 |

Rare and missing roles in dev:
* Dissent and PrecedentOverruled have **zero** dev sentences in both domains.
* PrecedentNotReliedUpon has none in CL dev (50 in IT).
* Issue (3 CL / 12 IT) and None (2 CL / 31 IT) are very rare.
## Method

### 1. Building one example per sentence
* **Pre-tokenisation.** Every sentence is tokenised once and cached; an empty sentence becomes `<unk>`.
* **Layout.**

  ```
  <s> [previous sentence] [TARGET sentence] [next sentence] </s>
  ```

  The target is labelled; context sentences are not.
* **384-token budget.**
  1. The target keeps up to 384 tokens (its last 384 if longer).
  2. The remaining budget is filled with the end of the previous sentence.
  3. Then the start of the next sentence.
* **Pooling mask.** A separate mask marks exactly the target sentence's token positions.

### 2. Encoder and classifier
* **Encoder.** The Phase-1 embeddings, all 16 blocks and the final RMSNorm are reused with a padding-only attention mask, so the encoder is **bidirectional**: the target attends to its context and vice versa. The LM head is not used.
* **Target-only pooling.** Hidden states are mean-pooled **only over the target sentence's tokens**. Context shapes the representation but does not dilute it.
* **Head.** Dropout 0.15, then `Linear(768 → 13)`.
* **Classification.** Each sentence is classified independently; there is no CRF or label-transition model across sentences.

### 3. Training loss
* **Total loss.** `L = CE + 0.01 · L_aux`: unweighted cross-entropy over the 13 roles plus the MoE balance loss.
* **Distillation (KD runs).** `L = 0.5 · (CE + 0.01 · L_aux) + 0.5 · T² · KL(p_teacher^T ‖ p_student^T)` with T = 2.
  * The teacher is the full-FT model of the same variant and context radius. It is trained or loaded automatically, kept frozen, and run online alongside the student.
  * The student copies the embeddings, Phase-1 blocks {0, 4, 8, 12} and the final norm (53.35M), and every parameter is trainable.

### 4. Domains
* **Training data.** CL and IT training documents are merged into one set (16,845 sentences), so one model serves both domains.
* **Evaluation.** Dev is scored on CL + IT combined (used for model selection) and on each domain separately (per-domain classification reports).

### 5. Fine-tuning methods
| Method | What is trained | Backbone LR | Epochs run |
|---|---|---|---|
| Full FT | all 152.14M parameters | 2e-5 | 4 (best: epoch 3) |
| LoRA, r = 8, α = 16 | `A, B` on attention projections + head (0.65M, 0.42 %) | 5e-4 | 3 (best: epoch 2) |
| LoRMA, r = 8, α = 16 | multiplicative `A, B` on attention outputs + head | 5e-4 | 3 |
| KD student | blocks {0, 4, 8, 12} + head, all trainable (53.35M) | 5e-5 | 3 |

* **Adapted layers.** LoRA and LoRMA wrap `q_proj, k_proj, v_proj, o_proj` (GQA) and `q_down, q_up, kv_down, kv_up, k_rope_proj, o_proj` (MLA). Experts and the shared expert are skipped.
* **Update rules.** LoRA: `y = W x + (α/r) B A x`. LoRMA: `y = W x + (α/r) B A (W x)`.
* **Initialisation.** `B` starts at zero, so training starts from the pretrained model.

### 6. Inference
Test documents are encoded with the same example builder and radius, and predictions are reordered by sentence index.
The writer checks three things for every document:
* one label per given sentence;
* every label is in the official vocabulary;
* CL and IT IDs don't collide.

### Fine-tuning techniques

| Method | Backbone LR | Epochs run | Total / trainable params |
|---|---|---|---|
| Full FT | 2e-5 | 4 (best: epoch 3) | 152.14M / 152.14M |
| LoRA (r = 8, α = 16, attention only) | 5e-4 | 3 (best: epoch 2) | 152.78M / 0.65M (0.42 %) |
| LoRMA (r = 8) | 5e-4 | 3 | ~0.65M trainable |
| KD student: blocks {0, 4, 8, 12}; loss = 0.5·CE + 0.5·T²·KL, T = 2; teacher = GQA-RoPE full FT | 5e-5 | 3 | 53.35M / 53.35M |

The head learning rate is 5e-4.
Other settings: batch size 16, AdamW (weight decay 0.01) and bf16 autocast.

## Results (local dev, Macro-F1 over all 13 labels)

| Run | Combined | CL | IT |
|---|---|---|---|
| **GQA-RoPE full FT** (epoch 3 of 4) | **0.3628** | 0.2866 | 0.3806 |
| GQA-RoPE LoRA | 0.3305 | 0.2900 | 0.3388 |
| GQA-RoPE KD student (53.35M) | 0.3128 | 0.2598 | 0.3287 |
| MLA-RoPE full FT | 0.3110 | 0.2546 | 0.3178 |
| MLA-Sin full FT  | 0.2900 | – | – |
| GQA-Sin LoRA | 0.2882 | 0.2195 | 0.3190 |



**Per-class F1 for the best model** (GQA-RoPE full FT, CL / IT):

| Role | F1 (CL / IT) |
|---|---|
| Statute | 0.84 / 0.70 |
| Fact | 0.49 / 0.67 |
| RatioOfTheDecision | 0.49 / 0.47 |
| ArgumentPetitioner | 0.46 / 0.34 |
| ArgumentRespondent | 0.34 / 0.31 |
| PrecedentReliedUpon | 0.30 / 0.43 |
| RulingByLowerCourt | 0.26 / 0.31 |

RatioOfTheDecision has high recall but low precision: recall 0.66 / 0.74, precision 0.39 / 0.35.

<!-- **Why local scores are lower than Codabench.** The local metric averages over all 13 labels, so Dissent and PrecedentOverruled (zero dev support) always contribute F1 = 0. This is likely part of why most Codabench scores are higher than local ones. -->

## Submissions

| Codabench file | Run | Local dev | Codabench |
|---|---|---|---|
| `submission_10.zip` | GQA-RoPE full FT (best upload) | – | **0.4128** |
| `submission_gqa_rope_moe_full_ctx3.zip` | GQA-RoPE full FT | 0.3628 | 0.4036 |
| `submission_gqa_rope_moe_lora_ctx3.zip` | GQA-RoPE LoRA | 0.3305 | 0.3854 |
| `submission_mla_rope_moe_full_ctx3.zip` | MLA-RoPE full FT | 0.3110 | 0.3418 |
| `submission_mla_sin_moe_full_ctx3.zip` | MLA-Sin full FT (1 epoch) | 0.2900 | 0.3399 |
| `submission_gqa_sin_moe_lora_ctx3.zip` | GQA-Sin LoRA | 0.2882 | 0.3197 |
| `submission_gqa_rope_moe_kd_ctx3.zip`  | GQA-RoPE KD student | 0.3128 | 0.3168 |

## How to run

1. Attach the `phase1-models` Kaggle dataset, add the `HF_TOKEN` secret, and run the Phase-1 setup cells (login, GPU, tokenizer and model definitions). The Phase-2 cells can be skipped.
2. In the "Phase 3" section, run the cells that copy `tokenizer.pkl` and `phase1-models`. Checkpoints are read from `/kaggle/working/phase1-models/best_models/abl_<variant>_best.pt`.
3. Run **CELL 1–5**:
   1. config;
   2. data download from `Phase_3_RR`;
   3. example building for radii 0–3;
   4. the model and adaptation methods;
   5. evaluation, checkpointing and training.
4. Train with `rr_run_experiment(variant, method, radius=3, epochs=...)`. For example:
   ```python
   rr_run_experiment("gqa_rope_moe", "full", radius=3, epochs=4)
   rr_run_experiment("gqa_rope_moe", "kd",   radius=3, epochs=3)   # trains/loads the full-FT teacher first
   rr_run_experiment("gqa_rope_moe", "lora", radius=3, epochs=3)
   ```
   Each run saves `rr_runs/<variant>__<method>__ctx3_best.pt` and `_last.pt` and resumes automatically. Run one experiment at a time: MLA full FT can exceed T4 memory.

5. Create submissions with `rr_submit_all`:
   ```python
   rr_submit_all([{"variant": "gqa_rope_moe", "method": "full", "radius": 3}])
   ```
   This writes `rr_runs/submission_<variant>_<method>_ctx3.zip`. It checks one valid label per sentence, unique document IDs across CL and IT, and that the zip contains only `prediction.json`.

## Outputs (`/kaggle/working/rr_runs/`)

* `<variant>__<method>__ctx3_best.pt` / `_last.pt`
* `submission_<variant>_<method>_ctx3.zip`, which contains `prediction.json`
* `phase3_rr_results.csv`
