# CS781 Capstone — In-House Legal Intelligence System

A self-hosted legal NLP stack trained **from scratch** on the official
[`Exploration-Lab/CS-781-Capstone`](https://huggingface.co/datasets/Exploration-Lab/CS-781-Capstone) data.
One small foundation model (Phase 1) is pretrained on Indian legal text and then fine-tuned for six downstream legal tasks (Phases 2–7).
No pretrained language model and no external legal corpus is used anywhere.

**Author:** Pritam Priyadarshi (230793), IIT Kanpur

---

## Results at a glance

| Phase | Task | Metric | Best Codabench score | Best system |
|---|---|---|---|---|
| 1 | Legal LM pretraining (512-word prompt → 128-word continuation) | BERTScore F1 | **0.8325** | GQA-RoPE-MoE, 100M tokens |
| 2 | Legal NER (character-offset spans, 12 types) | strict Macro-F1 | **0.5418** | GQA-RoPE, full fine-tuning, Viterbi decoding |
| 3 | Rhetorical role prediction (13 roles, CL + IT) | Macro-F1 | **0.4128** | GQA-RoPE, full fine-tuning |
| 4 | Court judgment prediction (binary) | Macro-F1 | **0.7706** | GQA-RoPE, full FT, head+tail 8 chunks |
| 5 | Legal statute identification (100 labels, multi-label) | Macro-F1 | **0.3974** | GQA-RoPE (clean), full FT, label-wise attention |
| 6 | Abstractive summarization | BERTScore F1 | **0.8421** | GQA-RoPE (clean), full FT, decoder-only |
| 7 | Prior case retrieval (top-7) | Micro-F1@7 | **0.3142** | GQA-RoPE (clean), full FT bi-encoder + BM25 hybrid |

Every phase folder lists **all** of its Codabench submissions with their local dev scores.
Codabench shortens long file names, so the phase tables use the full names written by the notebooks.

---

## Repository layout

```
.
├── README.md                     ← this file
├── requirements.txt
├── Phase1_Pretraining/           phase_1.ipynb + README.md
├── Phase2_LNER/                  phase_2.ipynb + README.md
├── Phase3_RR/                    phase_3.ipynb + README.md
├── Phase4_CJPE/                  phase_4.ipynb + README.md
├── Phase5_LSI/                   phase_5.ipynb + README.md
├── Phase6_Summarization/         phase_6.ipynb + README.md
└── Phase7_PCR/                   phase_7.ipynb + README.md
```

Each notebook is self-contained and keeps the cell outputs of the runs used in the report.

---

## The foundation model (Phase 1)

A pre-norm, decoder-only Transformer, written from scratch in PyTorch:

| Component | Setting |
|---|---|
| Layers / width / query heads | 16 / 768 / 8 |
| Normalisation, activation | RMSNorm (pre-norm), GELU, no biases |
| Embeddings | 30,000-token byte-level BPE (own tokenizer), tied input/output |
| FFN | Layers 0–1 dense (d_ff = 2048); layers 2–15 **Mixture-of-Experts**: 8 routed experts (top-2) + 1 shared expert, d_ff = 512 each, Switch-style load-balancing loss (coef. 0.01) |
| Attention (variant) | **GQA** (2 KV heads) **or** **MLA** (low-rank latent KV, decoupled RoPE) |
| Position (variant) | **RoPE** **or** additive **sinusoidal** |
| Max context | 2,048 tokens (trained on 1,024-token windows) |
| Size | GQA-RoPE 152.1M total / 86.1M active per token; MLA-RoPE 154.3M; GQA-Sin 152.1M; MLA-Sin 153.3M |

A code assertion keeps every configuration inside the 50M–300M parameter limit.

**Pretraining data:** 200M training tokens and 5M validation tokens, sampled 25/25/30/20 % from Acts / Supreme Court / High Courts / Tribunals.
All four variants were trained for 30M tokens under identical settings; GQA-RoPE was then continued to 100M tokens.

---

## How later phases reuse the foundation

```
                    ┌─► P2  L-NER      : token head + constrained BIO Viterbi
                    ├─► P3  RR         : sentence mean-pool + linear head
Phase-1 checkpoint ─┼─► P4  CJPE       : chunk encoder → pooling → MLP
(4 variants)        ├─► P5  LSI        : chunk encoder → label-wise attention
                    ├─► P6  Summ.      : causal LM, document → summary
                    └─► P7  PCR        : bi-encoder + BM25 hybrid
```

* **Phases 2–5 and 7** drop the LM head and run the blocks with a padding-only mask, so the causal decoder works as a **bidirectional encoder**.
* **Phase 6** keeps causal attention and the LM head.
* **Fine-tuning techniques** were implemented from scratch as `nn.Linear` wrappers:
  * full fine-tuning;
  * partial fine-tuning (top blocks only);
  * gradual unfreezing;
  * **LoRA**;
  * **LoRMA** (multiplicative low-rank);
  * **QLoRA** (own 4-bit NormalFloat storage);
  * bottleneck adapters;
  * prompt tuning;
  * **knowledge distillation** into a 4-of-16-block student (~53.5M parameters).

  LoRA, LoRMA and QLoRA wrap attention projections only; router and experts stay frozen.
  BitFit, AdaLoRA, IA³ and prefix tuning are implemented in some notebooks but were not trained to completion.

---

## Running the code

### Environment

All experiments ran on **Kaggle, 2× Tesla T4 (16 GB)**, one GPU per run, using fp16 (or bf16 autocast in Phases 2–3).
Install dependencies with:

```bash
pip install -r requirements.txt
```

### 1. Hugging Face access

The dataset is downloaded with `huggingface_hub`.
On Kaggle, add a secret named **`HF_TOKEN`** (Add-ons → Secrets); the first cell of every notebook reads it.
Outside Kaggle, set `HF_TOKEN` in the environment and replace the `kaggle_secrets` lines with `os.environ["HF_TOKEN"]`.

### 2. Phase-1 checkpoints for Phases 2–7

Phases 2–7 load the Phase-1 checkpoints and tokenizer from a Kaggle dataset (`phase1-models`) that is copied into `/kaggle/working/phase1-models`:

```
phase1-models/
├── tokenizer.pkl
└── best_models/                       ← Phases 2–4 read from here
    ├── abl_gqa_rope_moe_best.pt
    ├── abl_mla_rope_moe_best.pt
    ├── abl_gqa_sin_moe_best.pt
    ├── abl_mla_sin_moe_best.pt
    └── best_models/                   ← Phases 5–7 read from here
        ├── abl_clean_gqa_rope_moe_best.pt
        ├── abl_gqa_rope_moe_best.pt
        ├── abl_mla_rope_moe_best.pt
        ├── abl_gqa_sin_moe_best.pt
        └── abl_mla_sin_moe_best.pt
```

To reproduce:

1. Run Phase 1, which writes these files to `checkpoints/` and zips them as `best_models.zip`.
2. Upload them as a Kaggle dataset.
3. Attach the dataset to the notebook.
4. Uncomment the `!cp -rv ../input/.../phase1-models ./` cell at the start of the phase section.

### 3. Notebook structure

Every notebook starts with the same Phase-1 setup:

1. Hugging Face login and GPU check.
2. Text cleaner and BPE tokenizer.
3. Model definitions.

The phase-specific section follows.
Run the notebook top to bottom.
Phases 4–7 also have a `PROFILE` switch (`"fast" | "balanced" | "full"`) to trade run time for data size; the reported runs use `"full"`.

### 4. Interrupted sessions

Kaggle kills long sessions, so every trainer saves:

* a `*_last.pt` checkpoint (model, optimizer, scaler, RNG state and in-epoch position) every 40–200 steps;
* a separate `*_best.pt` checkpoint.

Re-running a cell resumes mid-epoch with the identical data order.
Finished experiments are cached and are not retrained.

### 5. Submissions

Each phase writes `prediction.json` and zips it as `submission_<tag>.zip` (the zip contains only `prediction.json`).
Before writing, assertions check the official ID set, label vocabulary and output format.

---

## Typical training times (T4)

| Phase | Run | Time |
|---|---|---|
| 1 | One 30M-token variant | ~52 min |
| 1 | Continuation to 100M tokens | ~113 min |
| 2 | Full FT | 39 min |
| 4 | Full FT, 3 epochs | 485.5 min |
| 5 | Full FT, 2 epochs | 250.8 min |
| 6 | Full FT | 54–134 min |
| 7 | One 4-epoch run | ~20 min |

---

## References

The method papers are listed in the report.
Main ones:

* LoRA (Hu et al., 2021)
* LoRMA (Bihany, Patel & Modi, 2025)
* QLoRA (Dettmers et al., 2023)
* adapters (Houlsby et al., 2019)
* prompt tuning (Lester et al., 2021)
* knowledge distillation (Hinton et al., 2015)
* GQA (Ainslie et al., 2023)
* MLA (DeepSeek-V2, 2024)
* RoPE (Su et al., 2021)
* Switch Transformers (Fedus et al., 2022)
* BM25 (Robertson & Zaragoza, 2009)
* BERTScore (Zhang et al., 2020)
