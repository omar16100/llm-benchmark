"""Test serial decode with the rotating cache that MTP uses."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--draft", type=Path)
    parser.add_argument("--prompt-ids", type=Path, default=Path(__file__).with_name("prompt.ids.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-size", type=int, choices=(2, 3), default=3)
    args = parser.parse_args()
    if not args.model.is_dir() or args.draft is not None and not args.draft.is_dir():
        parser.error("Use existing local model directories.")
    args.output.mkdir(parents=True, exist_ok=False)

    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.utils import load_tokenizer
    from mlx_vlm.generate.ar import generate_step
    from mlx_vlm.speculative import load_drafter
    from mlx_vlm.speculative.drafters import validate_drafter_compatibility
    from mlx_vlm.utils import load_model

    ids = json.loads(args.prompt_ids.read_text())
    model = load_model(args.model)
    tokenizer = load_tokenizer(args.model)
    draft = None
    if args.draft:
        draft, kind = load_drafter(str(args.draft), kind="mtp")
        validate_drafter_compatibility(model, draft, kind)
    cache = make_prompt_cache(model.language_model)
    assert all(getattr(item, "offset", 0) == 0 for item in cache)
    from types import SimpleNamespace
    from mlx_vlm.speculative.mtp import _buffer_mtp_target_cache
    _buffer_mtp_target_cache(cache, SimpleNamespace(config=SimpleNamespace(block_size=2)), 3)
    tokens = []
    generator = generate_step(mx.array([ids]), model, None, None, max_tokens=1024,
                              temperature=0, seed=1234, prompt_cache=cache,
                              prefill_step_size=1024, draft_model=draft, draft_kind="mtp",
                              draft_block_size=args.block_size if draft is not None else None)
    try:
        for token, _ in generator:
            if token in tokenizer.eos_token_ids:
                break
            tokens.append(int(token))
    finally:
        generator.close()
    digest = lambda values: hashlib.sha256(json.dumps(values).encode()).hexdigest()
    receipt = {"arm": "mtp" if draft is not None else "serial", "block_size": args.block_size,
               "input_tokens": len(ids), "output_tokens": len(tokens),
               "prompt_sha256": digest(ids), "token_sha256": digest(tokens),
               "versions": {name: importlib.metadata.version(name) for name in ("mlx", "mlx-lm", "mlx-vlm")},
               "model_config_sha256": hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest(),
               "custom_model_hooks": False, "cache_variant": "MTP buffered cache in serial decode"}
    (args.output / "tokens.json").write_text(json.dumps(tokens)+"\n")
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
