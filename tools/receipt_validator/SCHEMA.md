# Receipt contract, version 1

All files use UTF-8 JSON or JSONL, except `prompt.txt`.
The prompt hash uses the exact UTF-8 file bytes.
Token hashes use SHA256 over `json.dumps(ids).encode()` with Python defaults.
Each token ID is a nonnegative integer.
JSON values must be finite; unavailable rates use null.

## result.json

| Field | Required value |
| --- | --- |
| `schema_version` | Integer 1 |
| `status` | `completed`, `failed`, or `controlled_stop` |
| `engine` | Object with nonempty `name` and `version` strings |
| `model` | Object with nonempty `id` and `revision` strings |
| `warmup_tokens` | Nonnegative integer; use the producer's actual warmup count |
| `decode_tps_slope` | Recomputed slope or null |

A completed request also supplies these fields:

| Field | Required value |
| --- | --- |
| `target_tokens`, `input_tokens_reported`, `prefill_tokens_observed` | Integers equal to the prompt ID count |
| `generation_tokens` | Integer equal to the token trace length |
| `cached_input_tokens`, `cache_initial_tokens` | Recorded integer zero values |
| `first_token_seconds` | First output-token timestamp |
| `decode_seconds` | Last timestamp minus first timestamp |
| `token_sha256` | Hash of the output token ID list |

The optional `comparison` object supplies `token_ids_file`, `token_sha256`, and Boolean `exact_tokens`.
The reference file contains a token ID list.
The path is relative to the receipt directory and must stay inside it.
An unequal comparison can pass integrity if its Boolean result correctly reports the difference.

## Completed-request files

| File | Content |
| --- | --- |
| `prompt.ids.json` | Input token ID list |
| `prompt.txt` | Exact prompt bytes |
| `prompt.meta.json` | `prompt_tokens`, `ids_sha256`, `text_sha256`, and `needle_texts` |
| `tokens.jsonl` | One object per output token: `index`, `token`, `seconds` |
| `prefill.jsonl` | Objects with cumulative `done` counts |

Token positions start at 1 and have no gaps.
Token times must be nonnegative and monotonic; equal times are valid for a token batch.
Prefill counts must be monotonic and end at the full prompt count.
Each configured needle text must occur exactly once in the prompt.
An empty needle list disables this narrow prompt check.

## Qualification policy

The policy supplies `schema_version`, `context_tokens`, `min_output_tokens`, `min_decode_tps`, and `telemetry`.
Qualification requires valid receipt integrity and a completed request.
Context count must equal the policy count.
Output count and decode rate must meet their policy minimums.
Each telemetry key selects a field from `telemetry.json`.
Its rule contains `equals`, `min`, `max`, or a combination.
The telemetry rules do not validate a complete time series or prove thermal stability.
An empty telemetry object disables those checks.
The policy does not include an answer-quality requirement.

## Validator report

The report contains separate `integrity` and `qualification` objects with named checks.
`integrity.passed` requires every receipt check to pass.
`qualification.evaluated` is false when no policy is supplied.
The `rates` object reports `slope_tps`, `endpoint_tps`, and `warmup_tokens`.
The endpoint rate uses the first and last selected token times.
It is a diagnostic value; qualification uses the independently recomputed slope.
A telemetry rule requires its key to exist, even if its expected value is null.
