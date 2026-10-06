# Ling KDA adapter comparison

Target: [TensorFold/Ling-3.0-flash-MLX-4bit](https://huggingface.co/TensorFold/Ling-3.0-flash-MLX-4bit).
Revision: `cae62774492b589eea4aab2d6d36fb167a67d90c`.

The five edits in [adapter.patch](adapter.patch) restore the configured decay bound, reference L2 epsilon, and float32 beta logits before sigmoid.
Current stock mlx-lm already contains these calculations.
The model package selects the custom adapter through `model_file`.

The synthetic CPU comparison uses 32 heads, 128 dimensions, nine tokens, and two input scales.
Output differences were zero; cache and split-state comparisons passed their tolerances.
The engine is MLX 0.32.2 with mlx-lm `0.32.1.dev4+g53b9af378`.
The [validation receipt](validation.json) records the comparisons.

```sh
python check_layer.py --model-dir /path/to/existing/model
```

Full-model, BF16, PyTorch, and Metal equivalence remain unproved.
The separate output-gate cast also needs BF16 validation.
TensorFold Zig has no Ling backend at `88c424e`; its CLI stops at `UnsupportedRegex` for this tokenizer.
The absence of a Zig backend does not validate the Python adapter.
