"""Load the verified Ling adapter with explicit reference corrections."""
import hashlib
import sys
import types

SOURCE_SHA256 = "0f69b77f6953bad62b4e83df009e70f0a8115d5a6aa55d006dbfe3016e23e638"
PATCHES = [
    ("from mlx_lm.models.gated_delta import gated_delta_update",
     "from mlx_lm.models.gated_delta import gated_delta_update, normalize_qk"),
    ("        self.no_kda_lora = args.no_kda_lora",
     "        self.no_kda_lora = args.no_kda_lora\n        self.lower_bound = args.kda_lower_bound"),
    ("        q = (self.scale**2) * mx.fast.rms_norm(q, None, 1e-6)\n        k = self.scale * mx.fast.rms_norm(k, None, 1e-6)",
     "        q, k = normalize_qk(q, k, inv_scale=self.scale, eps=1e-6)"),
    ("        beta_logits = self.b_proj(x).reshape(batch, length, self.num_heads)",
     "        beta_logits = self.b_proj(x).astype(mx.float32).reshape(batch, length, self.num_heads)"),
    ("            use_kernel=not self.training,",
     "            use_kernel=not self.training,\n            lower_bound=self.lower_bound,"),
]


def adapter(model_path):
    path = model_path / "bailing_hybrid.py"
    original = path.read_bytes()
    if hashlib.sha256(original).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("Ling adapter source hash changed")
    source = original.decode()
    for old, new in PATCHES:
        if source.count(old) != 1:
            raise RuntimeError("Ling adapter patch has no unique source match")
        source = source.replace(old, new)
    module = types.ModuleType("eight_ling_reference_adapter")
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    module.source = source
    module.receipt = {
        "original_sha256": SOURCE_SHA256,
        "patched_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "patches": [{"old": old, "new": new} for old, new in PATCHES],
        "reference": "modeling_bailing_moe_v3.py and FLA 9f38d24980c46d46bd38614e743cdacd21906578",
        "note": "The local adapter applies the decay limit, reference L2 epsilon, and float32 beta sigmoid. Model weights stay unchanged.",
    }
    return module
