"""CPU checks for the optional production adapter numeric policy."""

import hashlib
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

from openjev_phase1 import core
from openjev_phase1.numeric_policy import (
    SCHEMA, install_fp32_softcap, resolve_sidecar, validate_sidecar,
)


MODEL = "google/gemma-4-12B-it"
REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"


def _policy():
    return {"schema": SCHEMA, "model": MODEL, "revision": REVISION,
            "final_softcap": 30.0, "softcap_precision": "fp32",
            "must_reinstall_lm_head_hook_for_scoring": True, "step": 250}


def test_policy_accepts_pilot_sidecar_and_rejects_mismatch(tmp_path):
    path = tmp_path / "numeric-policy.json"
    raw = json.dumps(_policy()).encode()
    path.write_bytes(raw)
    assert validate_sidecar(path, model=MODEL, revision=REVISION) == hashlib.sha256(raw).hexdigest()
    for key, value in (("schema", "unknown"), ("model", "other"),
                       ("revision", "0" * 40), ("final_softcap", 25.0),
                       ("softcap_precision", "native"),
                       ("must_reinstall_lm_head_hook_for_scoring", False)):
        path.write_text(json.dumps({**_policy(), key: value}))
        with pytest.raises(ValueError, match="unsupported or mismatched"):
            validate_sidecar(path, model=MODEL, revision=REVISION)
    path.write_text("{")
    with pytest.raises(ValueError, match="malformed JSON"):
        validate_sidecar(path, model=MODEL, revision=REVISION)


def test_local_missing_and_hub_snapshot_style_symlink_policy(tmp_path):
    assert resolve_sidecar(str(tmp_path), "local", local=True, cache_dir=None) is None
    blob = tmp_path / "blob"
    blob.write_text(json.dumps(_policy()))
    sidecar = tmp_path / "numeric-policy.json"
    sidecar.symlink_to(blob)
    assert resolve_sidecar(str(tmp_path), "local", local=True, cache_dir=None) == sidecar
    assert validate_sidecar(sidecar, model=MODEL, revision=REVISION) == hashlib.sha256(blob.read_bytes()).hexdigest()
    sidecar.unlink()
    sidecar.symlink_to(tmp_path / "elsewhere")
    with pytest.raises(ValueError, match="regular file"):
        resolve_sidecar(str(tmp_path), "local", local=True, cache_dir=None)


def test_hub_sidecar_uses_pinned_revision_and_only_verified_absence(tmp_path, monkeypatch):
    import huggingface_hub as hub
    import huggingface_hub.errors as errors

    path = tmp_path / "policy.json"
    path.write_text(json.dumps(_policy()))
    calls = []
    monkeypatch.setattr(hub, "try_to_load_from_cache", lambda *a, **k: None)

    def download(*args, **kwargs):
        calls.append((args, kwargs))
        return str(path)

    monkeypatch.setattr(hub, "hf_hub_download", download)
    assert resolve_sidecar("org/adapter", "a" * 40, local=False, cache_dir="/cache") == path
    assert calls == [(("org/adapter", "numeric-policy.json"),
                      {"revision": "a" * 40, "cache_dir": "/cache"})]
    monkeypatch.setattr(hub, "try_to_load_from_cache", lambda *a, **k: hub._CACHED_NO_EXIST)
    assert resolve_sidecar("org/adapter", "a" * 40, local=False, cache_dir="/cache") is None
    assert len(calls) == 1

    class Verified404(Exception):
        pass

    class CacheMiss(Exception):
        pass

    monkeypatch.setattr(hub, "try_to_load_from_cache", lambda *a, **k: None)
    monkeypatch.setattr(errors, "RemoteEntryNotFoundError", Verified404)
    monkeypatch.setattr(hub, "hf_hub_download", lambda *a, **k: (_ for _ in ()).throw(Verified404()))
    assert resolve_sidecar("org/adapter", "a" * 40, local=False, cache_dir="/cache") is None
    monkeypatch.setattr(hub, "hf_hub_download", lambda *a, **k: (_ for _ in ()).throw(CacheMiss()))
    with pytest.raises(CacheMiss):
        resolve_sidecar("org/adapter", "a" * 40, local=False, cache_dir="/cache")


class Gemma4UnifiedForCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"final_logit_softcapping": 30.0})()
        self.lm_head = torch.nn.Linear(1, 3, bias=False, dtype=torch.bfloat16)
        with torch.no_grad():
            self.lm_head.weight.copy_(torch.tensor([[120.0], [116.0], [112.0]], dtype=torch.bfloat16))

    def forward(self, x):
        return 30.0 * torch.tanh(self.lm_head(x) / 30.0)


def test_hook_preserves_fp32_gradient_and_is_idempotent():
    gradients = {}
    for name in ("native", "fp32"):
        model = Gemma4UnifiedForCausalLM()
        if name == "fp32":
            install_fp32_softcap(model)
            handle = model._openjev_fp32_softcap_hook
            install_fp32_softcap(model)
            assert model._openjev_fp32_softcap_hook is handle
            assert len(model.lm_head._forward_hooks) == 1
            baseline = model(torch.ones((1, 1, 1), dtype=torch.bfloat16))
            wrapper_hook = model.lm_head.register_forward_hook(
                lambda _module, _inputs, output: output.float())
            try:
                assert torch.equal(model(torch.ones((1, 1, 1), dtype=torch.bfloat16)), baseline)
            finally:
                wrapper_hook.remove()
            assert len(model.lm_head._forward_hooks) == 1
            assert model._openjev_fp32_softcap_hook is handle
        logits = model(torch.ones((1, 1, 1), dtype=torch.bfloat16))
        torch.nn.functional.cross_entropy(logits[0, 0].float()[None], torch.tensor([0])).backward()
        gradients[name] = model.lm_head.weight.grad.float().abs().sum().item()
        assert logits.dtype == (torch.bfloat16 if name == "native" else torch.float32)
    assert gradients["native"] == 0.0
    assert gradients["fp32"] > 0.0


def test_unsupported_class_or_cap_rejected():
    model = Gemma4UnifiedForCausalLM()
    model.config.final_logit_softcapping = 20.0
    with pytest.raises(ValueError, match="cap 30"):
        install_fp32_softcap(model)
    with pytest.raises(ValueError, match="supported Gemma"):
        install_fp32_softcap(torch.nn.Linear(1, 3))


def test_core_loader_installs_optional_policy_and_records_digest(tmp_path, monkeypatch):
    import transformers

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"test-weights")
    sidecar = adapter / "numeric-policy.json"
    sidecar.write_text(json.dumps(_policy()))

    config = SimpleNamespace(model_type="gemma4_unified", final_logit_softcapping=30.0,
                             use_cache=True)
    config.get_text_config = lambda: config
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **k: config)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr(core, "_gemma_text_checkpoint", lambda *a, **k: {})

    def from_pretrained(_cls, *args, **kwargs):
        model = Gemma4UnifiedForCausalLM()
        model.config = config
        return model, {}

    monkeypatch.setattr(Gemma4UnifiedForCausalLM, "from_pretrained", classmethod(from_pretrained),
                        raising=False)
    monkeypatch.setattr(core, "_native_model_class", lambda *a: Gemma4UnifiedForCausalLM)

    class Wrapped(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            self.config = base.config

        def get_base_model(self):
            return self.base

        def forward(self, x):
            return self.base(x)

    fake_peft = SimpleNamespace(
        PeftModel=SimpleNamespace(from_pretrained=lambda model, *a, **k: Wrapped(model)),
        __version__="test")
    monkeypatch.setitem(sys.modules, "peft", fake_peft)

    model, _, metadata = core.load_causal_model(
        MODEL, REVISION, adapter=str(adapter), adapter_revision="local", cache_dir=str(tmp_path))
    assert metadata["numeric_policy"] == "fp32-final-softcap-cap30"
    assert metadata["numeric_policy_sha256"] == hashlib.sha256(sidecar.read_bytes()).hexdigest()
    assert metadata["numeric_policy_precision"] == "fp32"
    assert model(torch.ones((1, 1, 1), dtype=torch.bfloat16)).dtype == torch.float32
    assert model._openjev_fp32_softcap_hook is not None

    sidecar.unlink()
    native, _, native_metadata = core.load_causal_model(
        MODEL, REVISION, adapter=str(adapter), adapter_revision="local", cache_dir=str(tmp_path))
    assert "numeric_policy" not in native_metadata
    assert not hasattr(native, "_openjev_fp32_softcap_hook")
    assert native(torch.ones((1, 1, 1), dtype=torch.bfloat16)).dtype == torch.bfloat16


def test_core_is_imported_from_this_checkout():
    expected = Path(__file__).resolve().parents[1] / "src/openjev_phase1/core.py"
    assert Path(core.__file__).resolve() == expected
