# Bengali OCR Fine-tuning — Experiment Plan & Decision Log

## Context
Goal: Fine-tune Gemma 4 on Bengali OCR data, evaluate against open-source community baselines, and document the process. Started 2026-04-19.

## Decision Log

### Decision 1: autoresearch is not suitable for this project (2026-04-19)
- **What**: User asked to use https://github.com/karpathy/autoresearch for research.
- **Finding**: autoresearch is an autonomous *pretraining* experimentation framework (iterates `train.py` on a GPT-style LM with NVIDIA GPU + Flash Attention 3). It does NOT do literature review, fine-tuning, vision/OCR, or experiment planning.
- **Alternative found**: [wadeKeith/autoresearch-qwen](https://github.com/wadeKeith/autoresearch-qwen) — a vision-capable fork that fine-tunes Qwen3-VL-4B on DocVQA. Has `mlx` branch for Apple Silicon. Directly adaptable for Bengali OCR with Gemma 4.
- **Also found**: [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) — 1459-star MLX port, but text-only pretraining.
- **Decision**: Clone both forks. Use `autoresearch-qwen` as the vision experiment framework, adapted for Gemma 4 + Bengali data.

### Decision 2: Gemma 4 vision fine-tuning has critical blockers (2026-04-19)
- **mlx-vlm** ([Blaizzy/mlx-vlm](https://github.com/Blaizzy/mlx-vlm)): Best path for Apple Silicon but has NaN gradient bug in Gemma 4 training ([PR #969](https://github.com/Blaizzy/mlx-vlm/pull/969)). Three sub-bugs identified:
  1. `-inf` in attention mask overflows in softmax backward → NaN
  2. `.item()` calls detach from computation graph
  3. `@mx.compile` on `ensure_fused_sdpa` blocks backward pass
- **Unsloth**: CUDA-only for training. MPS support is inference-only. Vision LoRA has zero-gradient bug (#5039).
- **HF Transformers + PEFT**: MPS backend unreliable for 31B-scale. No bitsandbytes on MPS.
- **TRL**: CUDA-targeted, OOM even on 8×H200.
- **Decision**: Fix mlx-vlm ourselves (all three bugs), then use it for Gemma 4 fine-tuning on M3 Ultra.

### Decision 3: Fix mlx-vlm NaN gradient bugs (2026-04-19)
Applied three patches to `/Users/macmini/projects/mlx-vlm/`:

**Fix 1** — `mlx_vlm/models/gemma4/vision.py:509`
```python
# Before: neg_inf = mx.array(float("-inf"), dtype=inputs_embeds.dtype)
# After:  neg_inf = mx.array(-1e4, dtype=inputs_embeds.dtype)
```
Rationale: -inf causes 0 × -inf = NaN in softmax backward. -1e4 is large enough to zero softmax probs but stays finite in float16 (max ~65504).

**Fix 2** — `mlx_vlm/models/gemma4/vision.py:526-531`
```python
# Before: .item() loop with per-sample concatenation
# After:  mask multiplication + fixed-length slice
valid_mask_expanded = mx.expand_dims(valid_mask, -1).astype(pooled.dtype)
hidden_states = (pooled * valid_mask_expanded)[:, :self.default_output_length, :]
```
Rationale: .item() calls mx.eval() (illegal in mx.compile) and detaches from gradient graph.

**Fix 3** — `mlx_vlm/models/base.py:386`
```python
# Before: @mx.compile decorator on ensure_fused_sdpa
# After:  removed (comment explains why)
```
Rationale: @mx.compile caches forward-only trace; backward pass info lost → NaN gradients.

**Status**: Patches applied. Sent to codex (gpt-5.4) for critique. Awaiting review output at `/Users/macmini/projects/codex/mlxvlm_nan_fix_review_19apr2026.txt`.

**Codex review returned** (`/Users/macmini/projects/codex/mlxvlm_nan_fix_review_19apr2026.txt`):

- **Fix 1 (-1e4)**: Approved. -1e4 is correct for float16 safety.
- **Fix 2 (mask multiply)**: **Rejected initial approach.** Codex found that `pooled * valid_mask_expanded[:, :default_output_length]` is NOT semantically equivalent for variable-length batches. Updated to boolean indexing: `hidden_states = pooled[valid_mask][None]` — preserves row-major packing across batches without `.item()`.
- **Fix 3 (@mx.compile)**: Approved as short-term fix, but **updated comment** — MLX docs say compile() should work with training, so the issue is likely version/interaction-specific, not fundamental. Should benchmark prefill throughput for inference regression.
- **logit_softcap**: Do NOT remove @mx.compile preemptively. Add a finite-gradient test first; only change if test fails.
- **Tests recommended**: tiny-config gradient test (batch_size=2, different aspect ratios), packing unit test, smoke step on SuperGemma4-26B with `--train-vision`.

### Decision 7: Codex OCR eval review applied (2026-04-20)

Codex (gpt-5.4) reviewed `/Users/macmini/projects/bengali-ocr-finetune/eval/metrics.py` and found 3 HIGH-priority bugs:

1. **Grapheme splitter missed ঁ/ং/ঃ (candrabindu, anusvara, visarga)** — these combine with the preceding orthographic unit. Also missed ZWJ/ZWNJ handling. Fixed: added all three + joiners to `is_combining` check.

2. **No Unicode NFC normalization** — canonically equivalent Bengali spellings (decomposed nukta forms, decomposed vowel signs) scored as errors. Fixed: added `normalize_bengali()` applying NFC + whitespace collapse before CER/GER.

3. **Data loader schema wrong** — dataset uses `image`/`text` top-level parquet fields, not the conversation format coded. Loader needs rewrite before baselines run. (Pending fix.)

Additional improvements applied:
- Added `strip_model_formatting()` — strips VLM output artifacts (code fences, backticks, common prefixes)
- Added corpus-level CER/WER alongside per-sample macro average (corpus-level more stable for mixed-length data)
- Tests expanded: 16 → 21 passing (triple conjuncts, candrabindu, visarga, NFC, formatting strip)

Full codex output: `/Users/macmini/projects/codex/bengali_ocr_eval_review_20apr2026.txt`

### Decision 8: Baseline engine selection (2026-04-20)

| Engine | Status | Bengali support | Notes |
|---|---|---|---|
| Tesseract | ✅ added | `lang=ben` + `script/Bengali` installed | CPU, standard baseline |
| EasyOCR | ✅ added | `bn` language pack | CPU, GPU optional |
| PaddleOCR (traditional) | ✅ added | `lang=bengali` | CPU, fast, good for Indian scripts |
| PaddleOCR VLM (PP-DocBee) | ❌ skipped | Chinese/English only | PaddlePaddle-only, no MPS |
| SeaLLion VLM | ❌ skipped | No Bengali (SEA languages only) | Not relevant |
| Qwen2.5-VL-7B | 📋 planned | Strong multilingual OCR | VLM baseline via mlx-vlm |

### Decision 9: Baseline results established (2026-04-20)

| Engine | CER (macro) | CER (corpus) | WER | GER | Samples/sec |
|---|---|---|---|---|---|
| **EasyOCR** | 0.182 | **0.153** | 0.474 | 0.258 | 16.5 |
| Tesseract | 0.770 | 0.722 | 0.949 | 0.864 | 48.3 |
| PaddleOCR | failed | — | — | — | — |

- **EasyOCR is the baseline to beat**: 15.3% corpus CER on 200 test samples from `rifathridoy/bengali-ocr-synthetic`.
- **Tesseract is catastrophic** on this synthetic data (72% CER) — font diversity in the dataset overwhelms its models.
- **PaddleOCR**: Bengali lang code neither `"bengali"` nor `"bn"` works in current PaddleOCR version. Dropped.
- Results at `/Users/macmini/projects/bengali-ocr-finetune/results/baselines.json`.

### Decision 10: SuperGemma4-26B-MLX-4bit lacks vision tower (2026-04-20)

SuperGemma4-26B-MLX-4bit-v2 is a **text-only** quantization — the MLX conversion stripped 211 vision_tower parameters. Cannot be used for vision fine-tuning.

**Pivot**: downloading `mlx-community/gemma-4-e4b-it-4bit` (Gemma 4 E4B, ~3 GB, full multimodal with vision tower) as the fine-tuning base. E4B has 4.5B effective params — much smaller and faster to iterate than 26B/31B while still being Gemma 4 architecture (validates our mlx-vlm fixes).

### Decision 11: Fine-tuning script created (2026-04-20)

`train/finetune_bengali_ocr.py` — LoRA fine-tuning of Gemma 4 on Bengali OCR:
- Config: rank=16, alpha=32, lr=1e-4, batch_size=1, max_steps=500
- Data: 5K train subsample (of 27K) for fast iteration
- Eval: CER/WER on 200 test samples
- Prompt: "এই ছবি থেকে বাংলা টেক্সট পড়ুন।" (Read Bengali text from this image)
- Saves adapters every 50 steps + training curve
- Submitted to codex for review.

### Decision 4: Bengali OCR datasets selected (2026-04-19)
| Dataset | Size | Type | Use |
|---|---|---|---|
| [rifathridoy/bengali-ocr-synthetic](https://huggingface.co/datasets/rifathridoy/bengali-ocr-synthetic) | 30K samples | Synthetic printed | Training (primary) |
| [BN-HTRd](https://data.mendeley.com/datasets/743k6dm543/1) | 788 pages / 108K words | Handwritten | Evaluation |
| [Bengali.AI Grapheme](https://www.kaggle.com/competitions/bengaliai-cv19) | 200K images | Grapheme components | Potential augmentation |
| [BCD3](https://bengaliai.github.io/bbocr) | 88.5K words | Printed multi-domain | Evaluation |

### Decision 5: Evaluation metrics (2026-04-19)
- **Primary**: CER (Character Error Rate) — most relevant for Bengali with complex conjunct characters
- **Secondary**: WER (Word Error Rate) — word-level accuracy
- **Tertiary**: Grapheme-level error rate (using BnGraphemizer tokenization per GraDeT-HTR methodology)
- **Not using**: BLEU (not standard for OCR)

### Decision 6: Baselines to compare against (2026-04-19)
| Baseline | CER | Source |
|---|---|---|
| bbOCR (Bengali.AI, APSIS-Net + YOLOv8) | ~0.19 improvement margin vs Tesseract | [arxiv 2308.10647](https://arxiv.org/abs/2308.10647) |
| GraDeT-HTR (GPT-2 decoder, 87M params) | SOTA Bengali handwritten | [arxiv 2509.18081](https://arxiv.org/abs/2509.18081) |
| Tesseract (Bengali) | baseline | — |
| EasyOCR (Bengali) | beats Tesseract on diverse images | [IEEE study](https://ieeexplore.ieee.org/document/10969286/) |
| swapnillo/Bangla-OCR-SFT (Qwen3-VL fine-tune) | no published CER | [HuggingFace](https://huggingface.co/swapnillo/Bangla-OCR-SFT) |

## Experiment Plan (Phase 1: Data + Pipeline)

1. Download `rifathridoy/bengali-ocr-synthetic` (30K)
2. Build CER/WER evaluation harness
3. Run baselines (Tesseract, EasyOCR) on a test split
4. Validate data pipeline with Qwen3-VL on mlx-vlm (works TODAY)

## Experiment Plan (Phase 2: Gemma 4 Fine-tuning)

1. Verify mlx-vlm NaN fix with Gemma 4 E4B (smaller, faster iteration)
2. LoRA fine-tune Gemma 4 E4B on Bengali OCR synthetic data
3. Evaluate CER/WER on BN-HTRd + BCD3 test sets
4. Scale to Gemma 4 31B if E4B results are promising
5. Compare against all baselines

## Experiment Plan (Phase 3: autoresearch Loop)

1. Adapt `autoresearch-qwen` to use Gemma 4 + Bengali OCR data
2. Define metric: 1 - CER (higher is better) on a held-out validation split
3. Let the autoresearch agent iterate on training hyperparameters autonomously
4. Document all experiments in `results.tsv`

## Repository structure (planned)
```
/Users/macmini/projects/bengali-ocr-finetune/
├── data/                  # downloaded datasets
├── eval/                  # CER/WER harness
├── baselines/             # Tesseract, EasyOCR scripts
├── train/                 # fine-tuning scripts (mlx-vlm LoRA)
├── autoresearch/          # adapted autoresearch-qwen loop
├── results/               # per-experiment outputs
├── docs/                  # this file + future findings
└── todo.md
```
