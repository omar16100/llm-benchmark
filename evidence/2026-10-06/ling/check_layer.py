"""Compare the corrected adapter with the stock KDA layer on the CPU."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.models.bailing_moe_v3 import BailingKDA, ModelArgs
from mlx_lm.models.cache import ArraysCache
import numpy as np

from ling_adapter import adapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    args = parser.parse_args()
    mx.set_default_device(mx.cpu)
    mx.random.seed(921)
    module = adapter(args.model_dir)
    config = json.loads((args.model_dir / "config.json").read_text())
    config.update(hidden_size=8, num_attention_heads=32, head_dim=128,
                  short_conv_kernel_size=4, no_kda_lora=True, kda_safe_gate=True,
                  kda_lower_bound=-5.0, rms_norm_eps=1e-6)
    corrected = module.KimiDeltaAttention(module.ModelArgs.from_dict(config))
    stock = BailingKDA(ModelArgs.from_dict(config))
    stock.load_weights([(name.replace("_conv1d.conv.", "_conv1d."), value)
                        for name, value in tree_flatten(corrected.parameters())])
    corrected.train(True)
    stock.train(True)
    x = mx.random.normal((1, 9, 8))
    checks = []
    for scale in (1.0, 0.001):
        value = x * scale
        left_cache, right_cache = ArraysCache(4), ArraysCache(4)
        left = corrected(value, cache=left_cache)
        right = stock(value, cache=right_cache)
        mx.eval(left, right, left_cache.cache, right_cache.cache)
        np.testing.assert_allclose(np.array(left), np.array(right), rtol=2e-5, atol=2e-6)
        for a, b in zip(left_cache.cache, right_cache.cache):
            np.testing.assert_allclose(np.array(a), np.array(b), rtol=2e-5, atol=2e-6)
        split_cache = ArraysCache(4)
        first = corrected(value[:, :4], cache=split_cache)
        second = corrected(value[:, 4:], cache=split_cache)
        split = mx.concatenate([first, second], axis=1)
        mx.eval(split, split_cache.cache)
        np.testing.assert_allclose(np.array(left), np.array(split), rtol=2e-5, atol=2e-6)
        for a, b in zip(left_cache.cache, split_cache.cache):
            np.testing.assert_allclose(np.array(a), np.array(b), rtol=2e-5, atol=2e-6)
        checks.append({"scale": scale, "stock_output_and_state": True,
                       "split_output_and_state": True,
                       "max_output_difference": float(mx.max(mx.abs(left-right)).item())})
    import mlx_lm.models.bailing_moe_v3 as reference
    print(json.dumps({"passed": True, "device": "CPU", "dtype": "float32",
                      "shape": [1, 9, 32, 128], "checks": checks,
                      "mlx": importlib.metadata.version("mlx"),
                      "mlx_lm": importlib.metadata.version("mlx-lm"),
                      "stock_source_sha256": hashlib.sha256(Path(reference.__file__).read_bytes()).hexdigest(),
                      "model_weights_used": False, "full_model_equivalence": False}, indent=2))


if __name__ == "__main__":
    main()
