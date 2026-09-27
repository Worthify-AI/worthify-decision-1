"""CPU checks for the public LoRA data contract and artifact gates."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from openjev_phase1 import task_lora


def row(identity="a", state="One state"):
    return {"id": identity, "state": state, "question": "Choose one",
            "options": [{"id": "yes", "description": "Yes"},
                        {"id": "no", "description": "No"}], "gold_option_id": "yes"}


def test_rows_require_gold_unique_ids_and_option_bounds(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(row()) + "\n")
    assert task_lora.read_rows(path, gold=True) == [row()]
    path.write_text(json.dumps({**row(), "gold_option_id": "unknown"}) + "\n")
    with pytest.raises(ValueError, match="gold_option_id"):
        task_lora.read_rows(path, gold=True)
    path.write_text((json.dumps(row()) + "\n") * 2)
    with pytest.raises(ValueError, match="duplicate row IDs"):
        task_lora.read_rows(path, gold=True)
    path.write_text(json.dumps({**row(), "options": [{"id": "one", "description": "One"}]}) + "\n")
    with pytest.raises(ValueError, match="2-16"):
        task_lora.read_rows(path, gold=False)


def test_splits_reject_id_group_and_normalized_state():
    first = row()
    with pytest.raises(ValueError, match="ID"):
        task_lora.check_splits([first], [row("a", "Other")])
    with pytest.raises(ValueError, match="group_id"):
        task_lora.check_splits([{**first, "group_id": "g"}], [{**row("b", "Other"), "group_id": "g"}])
    with pytest.raises(ValueError, match="decision content"):
        task_lora.check_splits([first], [row("b", " ONE   STATE ")])
    task_lora.check_splits([first], [row("b", "Other")])


def test_remote_base_requires_immutable_revision():
    with pytest.raises(ValueError, match="pinned 40-character"):
        task_lora.base_identity("org/model", "main")
    assert task_lora.base_identity("org/model", "a" * 40) == {
        "source": "org/model", "revision": "a" * 40, "kind": "hub-commit"}


def test_public_cli_rejects_local_bundles(tmp_path):
    with pytest.raises(ValueError, match="local bundles are not supported"):
        task_lora.base_identity(str(tmp_path), "a" * 64)


def test_missing_training_dependencies_have_install_guidance(monkeypatch):
    monkeypatch.setattr(task_lora.importlib.util, "find_spec", lambda _: None)
    with pytest.raises(RuntimeError, match=r"pip install -e '.\[train\]'"):
        task_lora.require_training_dependencies("nf4")


def test_manifest_rejects_changed_adapter_and_base(tmp_path):
    run = tmp_path / "run"
    adapter = run / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    (adapter / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": "org/model", "revision": "a" * 40}))
    base = task_lora.base_identity("org/model", "a" * 40)
    task_lora.write_json(run / "manifest.json", {
        "schema": task_lora.SCHEMA, "base": base,
        "numeric_policy": {"schema": "worthify-native-logits-v1"},
        "adapter_sha256": task_lora.sha(adapter / "adapter_model.safetensors"),
        "adapter_config_sha256": task_lora.sha(adapter / "adapter_config.json"),
        "numeric_policy_sha256": None})
    assert task_lora.verify_manifest(run, "org/model", "a" * 40)["base"] == base
    with pytest.raises(ValueError, match="base identity"):
        task_lora.verify_manifest(run, "org/model", "b" * 40)
    config = adapter / "adapter_config.json"
    config.write_text(json.dumps({"base_model_name_or_path": "org/model", "revision": None}))
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["adapter_config_sha256"] = task_lora.sha(config)
    (run / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Adapter config base identity"):
        task_lora.verify_manifest(run, "org/model", "a" * 40)
    (adapter / "adapter_model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash differs"):
        task_lora.verify_manifest(run, "org/model", "a" * 40)


def test_score_create_only_and_accuracy(tmp_path, monkeypatch):
    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps(row()) + "\n")
    output = tmp_path / "scores.jsonl"
    monkeypatch.setattr(task_lora, "base_identity", lambda *_: {"source": "m", "revision": "r"})
    monkeypatch.setattr(task_lora, "load_causal_model", lambda *_a, **_k: (object(), object(), {}))
    monkeypatch.setattr(task_lora, "numeric_policy", lambda *_: {"schema": "worthify-native-logits-v1"})
    monkeypatch.setattr(task_lora, "direct_score", lambda *_a: {
        "option_ids": ["yes", "no"], "option_logits": [1.0, 0.0], "probabilities": [0.73, 0.27]})
    args = SimpleNamespace(output=output, input=source, run=None, model="m", revision="r",
                           quantization="none", cache_dir=None, max_tokens=10)
    assert task_lora.score(args, evaluate=True)["accuracy"] == 1.0
    assert json.loads(output.read_text())["selected_option_id"] == "yes"
    with pytest.raises(FileExistsError):
        task_lora.score(args, evaluate=True)


def test_fresh_reload_rejects_logit_drift(tmp_path, monkeypatch):
    validation = tmp_path / "validation.jsonl"
    validation.write_text(json.dumps(row()) + "\n")
    reference = {"id": "a", "option_ids": ["yes", "no"],
                 "option_logits": [1.0, 0.0], "prompt_sha256": "prompt",
                 "selected_option_id": "yes"}
    manifest = {"base": {"source": "org/model", "revision": "a" * 40},
                "adapter_sha256": "b" * 64, "validation_sha256": task_lora.sha(validation),
                "quantization": "none", "max_tokens": 20,
                "numeric_policy": {"schema": "worthify-native-logits-v1"},
                "reload_reference": [reference]}
    monkeypatch.setattr(task_lora, "verify_manifest", lambda *_: manifest)
    monkeypatch.setattr(task_lora, "load_causal_model", lambda *_a, **_k: (object(), object(), {}))
    monkeypatch.setattr(task_lora, "numeric_policy", lambda *_: manifest["numeric_policy"])
    monkeypatch.setattr(task_lora, "observations", lambda *_: [{**reference, "option_logits": [1.01, 0.0]}])
    args = SimpleNamespace(run=tmp_path, model="org/model", revision="a" * 40,
                           cache_dir=None, validation=validation)
    with pytest.raises(ValueError, match="option logits differ"):
        task_lora.fresh_load(args)


def test_real_tiny_adapter_pins_base_and_reloads_with_matched_initialization(tmp_path):
    """CPU integration: real PEFT tensors, optimizer update, serialization, fresh base."""
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    from transformers import LlamaConfig, LlamaForCausalLM

    base = {"source": "org/pinned-base", "revision": "a" * 40, "kind": "hub-commit"}
    config = LlamaConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=2, vocab_size=32)
    config._name_or_path = base["source"]
    config._commit_hash = base["revision"]
    torch.manual_seed(1)
    first = LlamaForCausalLM(config)
    state = {name: tensor.clone() for name, tensor in first.state_dict().items()}
    first, _ = task_lora.configure_adapter(first, "none", base, 42)
    initial_hash = task_lora.adapter_tensor_sha256(first)

    torch.manual_seed(9876)
    other_base = LlamaForCausalLM(config)
    other_base, _ = task_lora.configure_adapter(other_base, "none", base, 42)
    assert task_lora.adapter_tensor_sha256(other_base) == initial_hash

    ids = torch.tensor([[1, 2, 3]])
    first.train()
    optimizer = torch.optim.AdamW((p for p in first.parameters() if p.requires_grad), lr=1e-3)
    loss = torch.nn.functional.cross_entropy(first(ids).logits[:, -1, :], torch.tensor([4]))
    loss.backward()
    optimizer.step()
    assert task_lora.adapter_tensor_sha256(first) != initial_hash
    first.eval()
    with torch.inference_mode():
        reference = first(ids).logits.clone()
    task_lora.save_adapter(first, tmp_path / "adapter", base)
    saved = json.loads((tmp_path / "adapter" / "adapter_config.json").read_text())
    assert saved["revision"] == base["revision"]
    assert saved["base_model_name_or_path"] == base["source"]

    fresh_base = LlamaForCausalLM(config)
    fresh_base.load_state_dict(state)
    loaded = peft.PeftModel.from_pretrained(fresh_base, tmp_path / "adapter", is_trainable=False)
    loaded.eval()
    with torch.inference_mode():
        torch.testing.assert_close(loaded(ids).logits, reference, rtol=0, atol=1e-6)
    loaded.peft_config["default"].revision = None
    with pytest.raises(ValueError, match="Adapter config base identity"):
        task_lora.save_adapter(loaded, tmp_path / "bad-adapter", base)
