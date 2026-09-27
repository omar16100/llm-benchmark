# Security dependency upgrades (27 Sep 2026)

## Context

Dependabot alerts were enabled on 27 Sep 2026 and reported 54 open alerts, all on the root
`uv.lock` (the only Python manifest in the repo besides `pyproject.toml`; there are no
`requirements*.txt` files or nested projects). Severity: critical 4, high 24, medium 21, low 5.

| Package | Locked | Alerts | First patched | Pulled in by |
|---|---|---|---|---|
| nltk | 3.9.4 | 29 | 3.10.3 (one alert has no fix) | rouge-score, via lm-eval |
| aiohttp | 3.13.5 | 14 | 3.14.3 | deepeval, fsspec[http] (requested by datasets and evaluate) |
| anyio | 4.13.0 | 2 | 4.14.2 | anthropic, httpx, openai |
| urllib3 | 2.6.3 | 2 | 2.7.0 | requests, sentry-sdk (deepeval) |
| transformers | 5.5.0 | 1 | 5.10.0 | direct dependency |
| pytest | 9.0.2 | 1 | 9.0.3 | direct dev dependency, deepeval |
| setuptools | 82.0.1 | 1 | 83.0.0 | deepeval, pytablewriter (lm-eval) |
| pydantic-settings | 2.13.1 | 1 | 2.14.2 | deepeval |
| idna | 3.11 | 1 | 3.15 | anyio, httpx, requests |
| lxml | 6.0.2 | 1 | 6.1.0 | sacrebleu (lm-eval) |
| sqlitedict | 2.1.0 | 1 | no fix | lm-eval |

Dependabot PR #3 (anyio 4.13.0 to 4.14.2) covered only the two anyio alerts.

## Changes

- `uv lock --upgrade-package` for each package above (uv 0.9.29, `requires-python = ">=3.11"`
  unchanged). Resolved versions: aiohttp 3.14.3, anyio 4.15.1, idna 3.20, lxml 6.1.3,
  nltk 3.10.3, pydantic-settings 2.15.0, pytest 9.1.1, setuptools 84.0.0, transformers 5.17.0,
  urllib3 2.8.0. Pulled along: tokenizers 0.23.2 and safetensors 0.8.0 (transformers),
  typing-extensions 4.16.0 (anyio 4.15.1 floor). New: defusedxml 0.7.1 (nltk 3.10 dependency).
- `pyproject.toml` floors raised for the two direct dependencies so a fresh resolve cannot
  pick a vulnerable version: `transformers>=5.10.0`, `pytest>=9.0.3` (dev group).
- lm-eval stays at 0.4.11 (the version the lm-eval setup in `todo.md` was done with).
  lm-eval 0.4.13 still depends on sqlitedict and rouge-score, so upgrading it would not
  remove either unfixed alert.
- Supersedes Dependabot PR #3.

## Impact

- Tests: `uv run --locked pytest tests/` gives 131 passed, 12 skipped before and after.
  The mlx-marked tests (`uv run --with mlx-lm pytest tests/test_bench_niah_mlx.py`, mlx-lm
  0.31.3) give 4 passed before and after.
- Kimi-Linear tokenization is unchanged by the transformers 5.5.0 to 5.17.0 (tokenizers
  0.22.2 to 0.23.2) jump. For every rung of the `bench_niah_mlx.py` ladder
  (`build_haystack(seed=20260706)`, chat template applied, tiktoken held constant), the token
  ids have the same SHA-256 under both stacks, and the counts (2,113, 8,113, 32,113, 131,186,
  262,258, 524,404, 1,048,692) match the prompt token column in
  `06072026_kimi_linear_1m_verification.md`. This covers the Kimi tokenizer only (from a local
  copy of its tokenizer files), not other models' tokenizers or generation timing.
  `tests/test_niah_haystack.py` with that tokenizer gives 8 passed, 1 failed under both
  versions. The failure (`test_chat_template_exists`) is pre-existing: transformers 5.x
  `apply_chat_template(tokenize=True)` returns a dict unless `return_dict=False` is passed.
  The harnesses are not affected because they get the tokenizer from mlx-lm, whose wrapper
  forces `return_dict=False`.
- nltk 3.9.4 to 3.10.3: the `rouge-score` stemmed ROUGE scores and Treebank word tokenization
  used by lm-eval tasks produce identical output on a spot check.
- Reproducibility: the README table is built from `data/runs_20apr2026.csv` (OpenAI-compatible
  client, no lm-eval) and does not depend on these packages. The Kimi reproduction steps in
  `06072026_kimi_linear_1m_verification.md` ask for a separate environment with `transformers<5`;
  the project lock was already on 5.5.0 before this change, and that instruction is left as is.
  mlx-lm loads its tokenizer through transformers `AutoTokenizer`, so the ladder parity above is
  the evidence that the upgrade does not change the Kimi prompts. No lm-eval results are
  published in this repo.
- Two alerts have no patched release and were dismissed as tolerable risk on 27 Sep 2026
  (reasons recorded on the alerts). Reopen them if these call paths change:
  - nltk GHSA-8mgp-746c-j5xp (model persistence APIs bypass `nltk.pathsec`, which nltk 3.10
    enforces by default): it needs a caller that passes untrusted paths to the affected
    `TransitionParser`, `AveragedPerceptron`, `PerceptronTagger` save/load, or
    `save_maxent_params` APIs. Nothing in this repo imports nltk. nltk is only reached through
    dependencies (rouge-score, lm-eval task code, deepeval scorers), and a search of the
    installed packages outside nltk finds no call to those APIs.
  - sqlitedict GHSA-g4r7-86gm-pgqc (pickle deserialization of stored values): lm-eval is the
    only user, and only opens a SqliteDict in `CachingLM`, which it creates only when a cache
    path is set (`--use_cache`, a config file, or `simple_evaluate(use_cache=...)`).
    `run_eval.py` sets none of these, so this runner leaves caching disabled and opens no
    SqliteDict database. Exploiting it would also need an attacker-written cache database on
    the local disk.
- Follow-up (not in this change): make `tests/test_niah_haystack.py` importable without
  `PYTHONPATH=.` and pass `return_dict=False` in `test_chat_template_exists`.
