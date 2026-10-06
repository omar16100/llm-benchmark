"""Test singleton target verification as a diagnostic numerical reference."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import types


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--prompt-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-size", type=int, choices=(2, 3), required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.utils import load_tokenizer
    from mlx_vlm.generate.ar import generate_step
    from mlx_vlm.speculative import load_drafter
    from mlx_vlm.speculative.drafters import validate_drafter_compatibility
    from mlx_vlm.utils import load_model

    model = load_model(args.model)
    tokenizer = load_tokenizer(args.model)
    draft, kind = load_drafter(str(args.draft), kind="mtp")
    validate_drafter_compatibility(model, draft, kind)
    calls = 0
    positions = []

    def verify_hidden(self, inputs, cache):
        nonlocal calls
        calls += 1
        positions.append(max(getattr(c, "offset", 0) for c in cache))
        shared = {}
        hidden = []
        for index in range(inputs.shape[1]):
            row = self.model(inputs[:, index:index+1], cache=cache,
                             shared_kv_sink=shared, skip_final_norm=True)
            mx.eval(row)
            hidden.append(row)
        return mx.concatenate(hidden, axis=1), shared

    model.language_model.speculative_verify_hidden = types.MethodType(verify_hidden, model.language_model)
    ids = json.loads(args.prompt_ids.read_text())
    cache = make_prompt_cache(model.language_model)
    tokens = []
    started = time.perf_counter()
    generator = generate_step(mx.array([ids]), model, None, None, max_tokens=1024,
                              temperature=0, seed=1234, prompt_cache=cache,
                              prefill_step_size=1024, draft_model=draft, draft_kind="mtp",
                              draft_block_size=args.block_size)
    try:
        for token, _ in generator:
            if token in tokenizer.eos_token_ids:
                break
            tokens.append(int(token))
    finally:
        generator.close()
    receipt = {"variant": "singleton_reference", "production_fix": False,
               "block_size": args.block_size, "verify_calls": calls,
               "verify_first_position": min(positions), "verify_last_position": max(positions),
               "input_tokens": len(ids), "output_tokens": len(tokens),
               "token_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
               "generation_seconds": time.perf_counter()-started,
               "peak_memory_bytes": mx.get_peak_memory()}
    (args.output / "tokens.json").write_text(json.dumps(tokens)+'\n')
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
