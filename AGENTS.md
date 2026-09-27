# Repository instructions

- Work from the repository root in an isolated environment installed with `pip install -e '.[test]'`.
- Run `pytest -q`, `(cd results/raw && sha256sum -c SHA256SUMS)`, `python benchmarks/verify_published.py`, `python benchmarks/verify_decision_1.py`, and `python scripts/verify_snapshot.py` before proposing a release.
- Keep benchmark outputs create-only. Do not alter existing prediction rows or claims to improve reported results.
- New claims need matching text-free row evidence, aggregate reports, checksums, method and chart updates. Preserve model/source revisions and negative outcomes.
- No weights, caches, credentials, third-party source text or private inputs belong in this repository.
- Update `SOURCE_PROVENANCE.json` snapshot hashes when files change; preserve original source hashes and mark adapted files accurately.
