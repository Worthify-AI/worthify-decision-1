"""Public, create-only task-LoRA workflow for pinned causal language models.

Train requires disjoint train/validation JSONL. Verify reloads the saved adapter
in a fresh model and checks validation logits. Score/evaluate use the same readout.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import re
from pathlib import Path

from .core import _pinned_source, load_causal_model, validate_row
from .direct import score as direct_score
from .numeric_policy import SCHEMA as POLICY_SCHEMA, install_fp32_softcap
from .training import _configure_lora, batch_loss, randomized_row

SCHEMA = "worthify-task-lora-v1"
TOLERANCE = 1e-4


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def read_rows(path: Path, *, gold: bool) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            validate_row(row)
            if gold and row.get("gold_option_id") not in {x["id"] for x in row["options"]}:
                raise ValueError("gold_option_id must name one declared option")
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            raise ValueError(f"{path}:{number}: {error}") from error
        rows.append(row)
    if not rows:
        raise ValueError(f"{path} has no rows")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path} contains duplicate row IDs")
    return rows


def check_splits(train: list[dict], validation: list[dict]) -> None:
    """Reject exact IDs, declared groups, and normalized decision content leakage."""
    def keys(rows):
        ids = {row["id"] for row in rows}
        groups = {row["group_id"] for row in rows if row.get("group_id")}
        content = {re.sub(r"\s+", " ",
            (row["state"] if isinstance(row["state"], str) else
             json.dumps(row["state"], sort_keys=True, ensure_ascii=False)).casefold()).strip()
            for row in rows}
        return ids, groups, content
    for label, left, right in zip(("ID", "group_id", "decision content"), keys(train), keys(validation)):
        if left & right:
            raise ValueError(f"Train/validation leakage through {label}")


def base_identity(model: str, revision: str) -> dict:
    local = _pinned_source(model, revision, "model")
    if local:
        raise ValueError("worthify-lora requires a Hugging Face model ID and immutable commit; local bundles are not supported")
    return {"source": model, "revision": revision, "kind": "hub-commit"}


def require_training_dependencies(quantization: str) -> None:
    packages = ["peft"] + (["bitsandbytes"] if quantization == "nf4" else [])
    missing = [name for name in packages if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(f"Missing {', '.join(missing)}; install training dependencies with pip install -e '.[train]' from the repository root")


def configure_adapter(model, quantization: str, base: dict, seed: int):
    import torch
    # Base loading may consume RNG differently. Pair adapter initialization itself.
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model, version = _configure_lora(model, quantization)
    config = model.peft_config["default"]
    config.base_model_name_or_path = base["source"]
    config.revision = base["revision"]
    return model, version


def adapter_tensor_sha256(model) -> str:
    import torch
    digest = hashlib.sha256()
    for name, tensor in sorted(model.named_parameters()):
        if ".lora_" not in name:
            continue
        tensor = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":")).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def save_adapter(model, path: Path, base: dict) -> None:
    config = model.peft_config["default"]
    if config.base_model_name_or_path != base["source"] or config.revision != base["revision"]:
        raise ValueError("Adapter config base identity differs")
    model.save_pretrained(path)


def numeric_policy(model) -> dict:
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if type(base).__name__ in {"Gemma4ForCausalLM", "Gemma4UnifiedForCausalLM"}:
        install_fp32_softcap(model)
        return {"schema": POLICY_SCHEMA, "softcap_precision": "fp32",
                "final_softcap": 30.0, "must_reinstall_lm_head_hook_for_scoring": True}
    return {"schema": "worthify-native-logits-v1", "softcap_precision": "native",
            "must_reinstall_lm_head_hook_for_scoring": False}


def selected(result: dict) -> str:
    index = max(range(len(result["option_ids"])), key=result["option_logits"].__getitem__)
    return result["option_ids"][index]


def observations(model, tokenizer, rows: list[dict], metadata: dict, limit: int) -> list[dict]:
    model.eval()
    output = []
    for row in rows:
        result = direct_score(model, tokenizer, row, metadata, limit)
        output.append({"id": row["id"], "option_ids": result["option_ids"],
                       "option_logits": result["option_logits"],
                       "prompt_sha256": result["prompt_sha256"],
                       "selected_option_id": selected(result)})
    return output


def accuracy(observed: list[dict], rows: list[dict]) -> float:
    return sum(x["selected_option_id"] == row["gold_option_id"]
               for x, row in zip(observed, rows, strict=True)) / len(rows)


def train(args) -> dict:
    import torch
    if args.output.exists():
        raise FileExistsError(args.output)
    if (args.seed < 0 or args.epochs < 1 or args.batch_size < 1 or args.max_tokens < 1
            or not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
        raise ValueError("seed must be nonnegative; epochs, batch-size, and max-tokens must be positive")
    base = base_identity(args.model, args.revision)
    train_rows, valid_rows = read_rows(args.train, gold=True), read_rows(args.validation, gold=True)
    check_splits(train_rows, valid_rows)
    require_training_dependencies(args.quantization)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Training requires exactly one visible CUDA GPU")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model, tokenizer, metadata = load_causal_model(base["source"], base["revision"],
        quantization=args.quantization, cache_dir=str(args.cache_dir) if args.cache_dir else None)
    model, peft_version = configure_adapter(model, args.quantization, base, args.seed)
    initial_adapter_sha256 = adapter_tensor_sha256(model)
    policy = numeric_policy(model)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoints = args.output / "checkpoints"
    checkpoints.mkdir()
    best = (-1.0, -1)
    history = []
    for epoch in range(args.epochs):
        model.train()
        ordered = list(train_rows)
        random.Random(f"{args.seed}:{epoch}").shuffle(ordered)
        optimizer.zero_grad(set_to_none=True)
        for index, row in enumerate(ordered):
            example = randomized_row(row, seed=args.seed, epoch=epoch)
            loss, _, _ = batch_loss(model, tokenizer, [example], args.max_tokens)
            group_size = min(args.batch_size, len(ordered) - (index // args.batch_size) * args.batch_size)
            (loss / group_size).backward()
            if (index + 1) % args.batch_size == 0 or index + 1 == len(ordered):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        observed = observations(model, tokenizer, valid_rows, metadata, args.max_tokens)
        score = accuracy(observed, valid_rows)
        checkpoint = checkpoints / f"epoch-{epoch:03d}"
        checkpoint.mkdir()
        save_adapter(model, checkpoint, base)
        history.append({"epoch": epoch, "validation_accuracy": score,
                        "adapter_sha256": sha(checkpoint / "adapter_model.safetensors")})
        if score > best[0]:
            best = (score, epoch)
    chosen = checkpoints / f"epoch-{best[1]:03d}"
    model.delete_adapter("default")
    model.load_adapter(str(chosen), adapter_name="default", is_trainable=False)
    model.set_adapter("default")
    reference = observations(model, tokenizer, valid_rows[:8], metadata, args.max_tokens)
    final = args.output / "adapter"
    final.mkdir()
    save_adapter(model, final, base)
    if policy["schema"] == POLICY_SCHEMA:
        write_json(final / "numeric-policy.json", {**policy, "model": base["source"], "revision": base["revision"]})
    manifest = {"schema": SCHEMA, "base": base, "model": metadata, "numeric_policy": policy,
                "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
                "learning_rate": args.learning_rate, "quantization": args.quantization,
                "max_tokens": args.max_tokens, "train_sha256": sha(args.train),
                "validation_sha256": sha(args.validation), "train_rows": len(train_rows),
                "validation_rows": len(valid_rows), "peft_version": peft_version,
                "initial_adapter_tensor_sha256": initial_adapter_sha256,
                "history": history, "selected_epoch": best[1], "validation_accuracy": best[0],
                "adapter_sha256": sha(final / "adapter_model.safetensors"),
                "adapter_config_sha256": sha(final / "adapter_config.json"),
                "numeric_policy_sha256": sha(final / "numeric-policy.json") if (final / "numeric-policy.json").exists() else None,
                "reload_reference": reference, "reload_tolerance": TOLERANCE}
    write_json(args.output / "manifest.json", manifest)
    return manifest


def verify_manifest(run: Path, model: str, revision: str) -> dict:
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("base") != base_identity(model, revision):
        raise ValueError("Run manifest base identity differs")
    adapter = run / "adapter"
    for name, key in (("adapter_model.safetensors", "adapter_sha256"),
                      ("adapter_config.json", "adapter_config_sha256"),
                      ("numeric-policy.json", "numeric_policy_sha256")):
        path = adapter / name
        expected = manifest.get(key)
        if (sha(path) if path.exists() else None) != expected or (name != "numeric-policy.json" and expected is None):
            raise ValueError(f"Adapter {name} hash differs")
    if manifest.get("numeric_policy", {}).get("schema") == POLICY_SCHEMA:
        from .numeric_policy import validate_sidecar
        validate_sidecar(adapter / "numeric-policy.json", model=manifest["base"]["source"], revision=revision)
    elif manifest.get("numeric_policy", {}).get("schema") != "worthify-native-logits-v1":
        raise ValueError("Unsupported numeric policy")
    config = json.loads((adapter / "adapter_config.json").read_text())
    if config.get("base_model_name_or_path") != model or config.get("revision") != revision:
        raise ValueError("Adapter config base identity differs")
    return manifest


def fresh_load(args, *, verify: bool = True):
    manifest = verify_manifest(args.run, args.model, args.revision)
    model, tokenizer, metadata = load_causal_model(manifest["base"]["source"], manifest["base"]["revision"],
        adapter=str(args.run / "adapter"), adapter_revision=manifest["adapter_sha256"],
        quantization=manifest["quantization"], cache_dir=str(args.cache_dir) if args.cache_dir else None)
    if numeric_policy(model) != manifest["numeric_policy"]:
        raise ValueError("Fresh model numeric policy differs")
    if verify:
        rows = read_rows(args.validation, gold=True)
        if sha(args.validation) != manifest["validation_sha256"]:
            raise ValueError("Validation file hash differs")
        current = observations(model, tokenizer, rows[:8], metadata, manifest["max_tokens"])
        reference = manifest["reload_reference"]
        if len(current) != len(reference):
            raise ValueError("Fresh reload reference row count differs")
        for one, two in zip(reference, current, strict=True):
            if any(one[key] != two[key] for key in ("id", "option_ids", "prompt_sha256", "selected_option_id")):
                raise ValueError("Fresh reload decision or prompt differs")
            if len(one["option_logits"]) != len(two["option_logits"]) or any(
                not math.isfinite(a) or not math.isfinite(b) or abs(a - b) > TOLERANCE
                for a, b in zip(one["option_logits"], two["option_logits"], strict=True)):
                raise ValueError("Fresh reload option logits differ")
    return model, tokenizer, metadata, manifest


def verify(args) -> dict:
    if args.output and args.output.exists():
        raise FileExistsError(args.output)
    _, _, metadata, manifest = fresh_load(args)
    receipt = {"schema": SCHEMA + "-verification", "base": manifest["base"],
               "adapter_sha256": manifest["adapter_sha256"],
               "validation_sha256": manifest["validation_sha256"],
               "checked_rows": len(manifest["reload_reference"]),
               "tolerance": TOLERANCE, "model": metadata}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, receipt)
    return receipt


def score(args, *, evaluate: bool) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = read_rows(args.input, gold=evaluate)
    if args.run:
        model, tokenizer, metadata, manifest = fresh_load(args, verify=True)
        limit = manifest["max_tokens"]
    else:
        base = base_identity(args.model, args.revision)
        model, tokenizer, metadata = load_causal_model(base["source"], base["revision"],
            quantization=args.quantization, cache_dir=str(args.cache_dir) if args.cache_dir else None)
        metadata["numeric_policy"] = numeric_policy(model)
        limit = args.max_tokens
    args.output.parent.mkdir(parents=True, exist_ok=True)
    correct = 0
    with args.output.open("x") as stream:
        for row in rows:
            item = direct_score(model, tokenizer, row, metadata, limit)
            item["selected_option_id"] = selected(item)
            if evaluate:
                item["gold_option_id"] = row["gold_option_id"]
                correct += item["selected_option_id"] == row["gold_option_id"]
            stream.write(json.dumps(item, sort_keys=True, allow_nan=False) + "\n")
    return {"rows": len(rows), "accuracy": correct / len(rows) if evaluate else None,
            "output": str(args.output), "probability_status":
            "conditional option score; uncalibrated as decision confidence"}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "verify", "evaluate", "score"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--model", required=True)
        cmd.add_argument("--revision", required=True)
        cmd.add_argument("--cache-dir", type=Path)
        if name == "train":
            cmd.add_argument("--train", type=Path, required=True)
            cmd.add_argument("--validation", type=Path, required=True)
            cmd.add_argument("--output", type=Path, required=True)
            cmd.add_argument("--seed", type=int, required=True)
            cmd.add_argument("--epochs", type=int, required=True)
            cmd.add_argument("--batch-size", type=int, default=16)
            cmd.add_argument("--learning-rate", type=float, default=2e-4)
            cmd.add_argument("--quantization", choices=("none", "nf4"), default="nf4")
            cmd.add_argument("--max-tokens", type=int, default=2048)
        elif name == "verify":
            cmd.add_argument("--run", type=Path, required=True)
            cmd.add_argument("--validation", type=Path, required=True)
            cmd.add_argument("--output", type=Path)
        else:
            cmd.add_argument("--input", type=Path, required=True)
            cmd.add_argument("--output", type=Path, required=True)
            cmd.add_argument("--run", type=Path)
            cmd.add_argument("--validation", type=Path)
            cmd.add_argument("--quantization", choices=("none", "nf4"), default="none")
            cmd.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args(argv)
    if args.command in ("evaluate", "score") and args.run and not args.validation:
        parser.error("--run requires --validation for fresh reload verification")
    result = train(args) if args.command == "train" else verify(args) if args.command == "verify" else score(args, evaluate=args.command == "evaluate")
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
