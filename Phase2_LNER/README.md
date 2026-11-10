# Phase 2 — Legal Named Entity Recognition (L-NER)

**Notebook:** `phase_2.ipynb`

## Task

| | |
|---|---|
| Input | Full text of an Indian judgment |
| Output | `{doc_id: [[start, end, LABEL, text], ...]}`, character offsets into the original string |
| Labels (12) | APP, RESP, A.COUNSEL, R.COUNSEL, JUDGE, WIT, AUTH, COURT, STAT, PREC, DATE, CASENO |
| Metric | strict Macro-F1: a span counts only if start, end **and** label all match exactly |

## Data

* **Training data:** labelled folds 1 and 2 (35 documents each) are pooled and split 80/20 by document into **56 train / 14 validation** (seed 42).
* **Test set:** `fold3`.
* **Validation label counts:** DATE 315, STAT 212, PREC 201, AUTH 190, CASENO 161, COURT 136, RESP 66, JUDGE 64, WIT 62, APP 61, A.COUNSEL 26, R.COUNSEL 21.
* **Offset oracle:** after tokenisation, 4,049 of 4,120 gold entities (98.3 %) start and end on token boundaries. Decoding gold BIO tags back to spans gives an upper bound of **0.978** strict Macro-F1 on validation. The lowest per-label bounds are PREC 0.876, APP 0.934 and STAT 0.948.

## Method

### 1. Offset-preserving tokenisation
* **No text changes.** The judgment text is never cleaned or lowercased, so every predicted offset indexes directly into the original string.
* **Offsets per token.** `encode_with_offsets` replays the tokenizer's own pre-tokenisation regex and ranked BPE merges word by word. It carries a byte-to-character map, so every BPE token gets exact `[start, end)` character offsets; this is also safe for multi-byte UTF-8 characters. Leading whitespace is stripped from each token's span.
* **Gold spans to BIO tags.** Every token that overlaps a gold entity is labelled: the first gets `B-X` and the rest get `I-X`.
  * 98.3 % of entities line up exactly with token boundaries.
  * Decoding the gold tags back to spans gives an upper bound of **0.978** strict Macro-F1 on validation.

### 2. Long documents: sliding windows
* **Training.** Each document's token sequence is cut into 512-token windows with stride 256, and the last window is aligned to the end of the document. Each window is prefixed with `<s>`, which is ignored by the loss (label −100).
* **Inference.** The same windows are used (stride `WIN // 2` = 256). Every token's log-softmax scores are averaged over all windows that cover it before decoding.

### 3. Encoder and tagging head
* **Encoder.** The Phase-1 token embeddings, all 16 blocks and the final RMSNorm are reused; the LM head is bypassed. The attention mask is padding-only, so the causal decoder runs as a **bidirectional encoder**.
* **Sinusoidal variants.** Embeddings are scaled by √d and the sinusoidal table is added, exactly as in Phase 1.
* **MoE loss.** Each block's MoE load-balancing loss is summed and added to the training loss with weight 0.01.
* **Head.** Dropout 0.1, then `Linear(768 → 25)` over the BIO tag set (O plus B-/I- for 12 types). An MLP head is also implemented but was not used.

### 4. Training loss
* **Class weights.** Token-level weighted cross-entropy with `w_c = min(10, ((n_O + 1) / (n_c + 1))^0.5)` and `w_O = 1`. This up-weights rare entity tags without letting any weight exceed 10.
* **Total loss.** `L = CE_weighted + 0.01 · L_aux`.
* **Distillation (KD runs).** `L = 0.5 · (CE + 0.01 · L_aux) + 0.5 · T² · KL(p_teacher^T ‖ p_student^T)` with T = 2, computed over real (non-ignored) tokens. The teacher is the frozen full-FT model of the same variant and runs online alongside the student.

### 5. Decoding
* **Constrained Viterbi** over the averaged log-probabilities. Transitions into `I-X` are allowed only from `B-X` or `I-X`, and a sequence may not start with `I-`.
* **Tags to spans.** A span starts at its first token's start offset and ends at its last token's end offset; a change of entity type starts a new span. By construction, predicted spans never overlap or nest.
* **Ablation.** Greedy argmax tagging is kept for comparison: 0.4196 versus 0.5421 for Viterbi with the same model.


### 6. Fine-tuning methods
| Method | What is trained | Backbone LR | Epochs |
|---|---|---|---|
| Full FT | all ~152M parameters + head | 5e-5 | 13 |
| LoRA, r = 8, α = 16 | low-rank `A, B` on attention projections + head (0.65–0.94M) | 1e-3 | 10 |
| LoRMA, r = 8, α = 16 | multiplicative low-rank update on attention outputs + head (0.65–0.92M) | 1e-3 | 10 |
| KD student | embeddings + Phase-1 blocks {0, 2, 4, 7} + head, all trainable (53.4–53.9M) | 1e-4 | 10 |

* **LoRA:** `y = W x + (α/r) · B A x`.
* **LoRMA:** `y = (I + (α/r) · B A) W x`, applied to the output of the frozen projection.

In both, `B` is initialised to zero, so training starts from the pretrained model's behaviour. Experts and the router stay frozen.
Partial FT (top 3 blocks) and bottleneck adapters (64-d) are implemented but were not submitted.
### Fine-tuning techniques compared

| Method | Trainable | Backbone LR | Epochs |
|---|---|---|---|
| Full fine-tuning | all (~152M) | 5e-5 | 13 |
| LoRA (r = 8, α = 16, attention projections only) | 0.65–0.94M | 1e-3 | 10 |
| LoRMA (multiplicative, r = 8) | 0.65–0.92M | 1e-3 | 10 |
| KD student: Phase-1 blocks {0, 2, 4, 7}, T = 2, α = 0.5, teacher = full FT of the same variant | 53.4–53.9M | 1e-4 | 10 |

The head learning rate is 1e-3, with AdamW (weight decay 0.01), batch size 4 and bf16 autocast.
Partial FT and adapters are implemented (`partial_top3`, `adapter64`) but were not submitted.

## Results

Best model: GQA-RoPE full fine-tuning.

* **Overall:** strict Macro-F1 **0.5421** on local validation (P 0.515, R 0.577) and **0.5418** on Codabench.
* **Training curve:** best at epoch 9; it declines to 0.4793 by epoch 13.
* **Decoding ablation:** greedy argmax tags give 0.4196 and constrained Viterbi gives 0.5421, with the same model.
* **Per-label F1:**

  | Label | F1 | Label | F1 |
  |---|---|---|---|
  | DATE | 0.949 | A.COUNSEL | 0.538 |
  | COURT | 0.831 | WIT | 0.371 |
  | STAT | 0.753 | RESP | 0.369 |
  | JUDGE | 0.689 | PREC | 0.314 |
  | R.COUNSEL | 0.609 | AUTH | 0.294 |
  | CASENO | 0.573 | APP | 0.216 |

* **Error breakdown** (validation): 964 exact, 432 false positives with no overlap, 250 boundary errors (same type, overlapping), 222 missed, 75 type confusions.

## Submissions

| Codabench file | Variant | Method | Local val | Codabench |
|---|---|---|---|---|
| `submission_gqa_rope_moe__full.zip` | GQA-RoPE | full FT | 0.5421 | **0.5418** |
| `submission_gqa_rope_moe__lorma.zip` | GQA-RoPE | LoRMA | 0.4986 | 0.4964 |
| `submission_gqa_rope_moe__lora.zip` | GQA-RoPE | LoRA | 0.5082 | 0.4929 |
| `submission_mla_rope_moe__full.zip` | MLA-RoPE | full FT | 0.5098 | 0.4755 |
| `submission_mla_rope_moe__lora.zip` | MLA-RoPE | LoRA | 0.4796 | 0.4414 |
| `submission_mla_rope_moe__kd.zip` | MLA-RoPE | KD student (53.9M) | 0.4393 | 0.4389 |
| `submission_mla_rope_moe__lorma.zip` | MLA-RoPE | LoRMA | 0.4902 | 0.4388 |
| `submission_gqa_sin_moe__lora.zip` | GQA-Sin | LoRA | 0.2678 | 0.2747 |
| `submission_gqa_sin_moe__lorma.zip` | GQA-Sin | LoRMA | 0.2875 | 0.2643 |
| `submission_mla_sin_moe__lora.zip` | MLA-Sin | LoRA (5 epochs) | 0.2404 | 0.2372 |

## How to run

1. Attach the `phase1-models` Kaggle dataset and add the `HF_TOKEN` secret.
2. Run the setup section (login, tokenizer, model definitions).
3. Run the `!cp -rv ../input/.../phase1-models/ ./` cell. Checkpoints are read from `/kaggle/working/phase1-models/best_models/abl_<variant>_best.pt`.
4. Run the "Phase 2 - L-NER" section top to bottom:
   1. configuration;
   2. data loading and the 80/20 document split;
   3. offset alignment and the oracle check;
   4. model builders;
   5. training.
5. Run the experiment cells. `run_variant_suite(...)` and the per-method runners train, evaluate and write the test predictions for each (variant, method). The best epoch is kept in `lner_runs/<variant>__<method>_best.pt`, and runs resume from `_last.pt`.
6. Use the saved-model loader cell to regenerate a submission from any `_best.pt` without retraining.

## Outputs (`/kaggle/working/lner_runs/`)

* `<variant>__<method>_best.pt` / `_last.pt`
* `prediction_<variant>__<method>.json`
* `submission_<variant>__<method>.zip`
