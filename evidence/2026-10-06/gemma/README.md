# Gemma MTP reproduction

The full Gemma 4 26B-A4B QAT model reproduces [mlx-vlm issue 2431](https://github.com/Blaizzy/mlx-vlm/issues/2431).
The machine is a 512 GB M3 Ultra.
The engine is mlx-vlm 0.7.4 with MLX 0.32.2 and mlx-lm `0.32.1.dev4+g53b9af378`.
Model revisions and configuration hashes are in [fresh-evidence.json](fresh-evidence.json).

Three stock API serial controls match.
Both MTP block sizes first differ at output index 177: serial token 2132, MTP token 5250.
The input has 2,048 tokens; the output has 1,024 tokens.
Three stock CLI serial answers also match; both MTP answers first differ at character 315.
The CLI comparison excludes only its status lines and final whitespace.

A serial control with post-prefill cache conversion also differs at index 177.
It differs from stock MTP at index 180, so it isolates the first flip only.
A singleton reference with serial ring layout matches all 1,024 output tokens for both block sizes.
Both block sizes also match a fresh 1,152-token serial control across the next cache boundary.
The first block-three reference uses the original script; the second adds only counters.

The reference is diagnostic code.
Its drafter receives ring-order KV although the drafter expects temporal order.
Its cache subclass lacks fields for unused inherited methods.
These limitations exclude production and throughput claims.
The separate effect of stock block arithmetic remains unmeasured.

```sh
python3 check_evidence.py
python stock_tokens.py --model /path/to/model --output serial
python stock_tokens.py --model /path/to/model --draft /path/to/assistant --block-size 3 --output mtp
python boundary_ring_probe.py --model /path/to/model --draft /path/to/assistant --prompt-ids prompt.ids.json --block-size 3 --output reference
```

The boundary reference uses 1,152 output tokens; compare it with `boundary_stock_tokens.py`.
The saved evidence check verifies 19 cases without MLX or model weights.
No 256K or speed qualification follows from these tests.
