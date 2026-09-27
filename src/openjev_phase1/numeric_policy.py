"""Optional, pinned FP32 final-softcap policy for adapter inference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA = "openjev-stability-pilot-numeric-policy-v1"
POLICY_NAME = "fp32-final-softcap-cap30"
SIDECAR = "numeric-policy.json"


def resolve_sidecar(adapter: str, revision: str, *, local: bool,
                    cache_dir: str | None) -> Path | None:
    """Resolve only the optional sidecar at the adapter's pinned revision."""
    if local:
        path = Path(adapter) / SIDECAR
        if not path.exists() and not path.is_symlink():
            return None
        if not path.is_file():
            raise ValueError("Adapter numeric-policy sidecar must be a regular file")
        return path
    from huggingface_hub import _CACHED_NO_EXIST, hf_hub_download, try_to_load_from_cache
    from huggingface_hub.errors import RemoteEntryNotFoundError

    cached = try_to_load_from_cache(adapter, SIDECAR, revision=revision,
                                    cache_dir=cache_dir)
    if cached is _CACHED_NO_EXIST:
        return None
    if isinstance(cached, str):
        return Path(cached)

    try:
        return Path(hf_hub_download(adapter, SIDECAR, revision=revision,
                                    cache_dir=cache_dir))
    except RemoteEntryNotFoundError:
        return None


def validate_sidecar(path: Path, *, model: str, revision: str) -> str:
    """Return the sidecar digest or fail closed on every unsupported policy."""
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        policy = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Adapter numeric-policy sidecar is malformed JSON") from error
    expected = {"schema": SCHEMA, "model": model, "revision": revision,
                "final_softcap": 30.0, "softcap_precision": "fp32",
                "must_reinstall_lm_head_hook_for_scoring": True}
    if not isinstance(policy, dict) or any(policy.get(key) != value for key, value in expected.items()):
        raise ValueError("Adapter numeric-policy sidecar is unsupported or mismatched")
    if isinstance(policy.get("final_softcap"), bool):
        raise ValueError("Adapter numeric-policy final softcap must be numeric")
    return digest


def install_fp32_softcap(model) -> None:
    """Cast LM-head output before Gemma's native divide/tanh/multiply."""
    if getattr(model, "_openjev_fp32_softcap_hook", None) is not None:
        return
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if type(base).__name__ not in {"Gemma4ForCausalLM", "Gemma4UnifiedForCausalLM"}:
        raise ValueError("FP32 softcap policy requires a supported Gemma 4 causal LM")
    cap = getattr(base.config, "final_logit_softcapping", None)
    if isinstance(cap, bool) or cap != 30.0 or not hasattr(base, "lm_head"):
        raise ValueError("FP32 softcap policy requires Gemma final cap 30 and lm_head")

    def upcast(_module, _inputs, output):
        return output.float()

    model._openjev_fp32_softcap_hook = base.lm_head.register_forward_hook(upcast)
