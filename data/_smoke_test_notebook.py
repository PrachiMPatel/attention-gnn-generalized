"""Smoke-test the bits we can run without torch/transformers.

Validates:
  - wits_main.ipynb and wits_binary_main.ipynb are valid JSON and have
    the expected cells.
  - wits_eval_cases.jsonl is well-formed and matches the WITS schema.
  - wits_eval_cases_binary.jsonl is well-formed.
  - the stratified-split logic in section 2 runs end-to-end on pure
    Python for both notebooks.

Intentionally skips any cell that imports torch / transformers /
torch_geometric — those run in the user's notebook environment.
"""
from __future__ import annotations

import ast
import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _check_notebook(nb_path: Path) -> None:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4, nb["nbformat"]
    assert len(nb["cells"]) > 10, f"unexpectedly few cells in {nb_path.name}: {len(nb['cells'])}"
    md_cells = [c for c in nb["cells"] if c["cell_type"] == "markdown"]
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    print(f"{nb_path.name}: {len(md_cells)} md + {len(code_cells)} code = {len(nb['cells'])} cells")
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if not src.strip():
            continue
        try:
            ast.parse(src)
        except SyntaxError:
            print(f"\nSYNTAX ERROR in {nb_path.name} cell[{i}]:")
            print(src)
            raise


def _check_4class_dataset() -> list[dict]:
    data = REPO / "data" / "wits_eval_cases.jsonl"
    valid = {"safe", "maybe_safe", "unsafe", "extremely_unsafe"}
    records = []
    for line in data.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        assert "command" in r and "verdict" in r and "shell" in r, r
        assert r["verdict"] in valid, r
        assert r["shell"] in ("bash", "powershell"), r
        records.append(r)
    print(f"\n4-class dataset OK: {len(records)} valid records")
    print("  by verdict:", dict(Counter(r["verdict"] for r in records)))
    print("  by shell  :", dict(Counter(r["shell"] for r in records)))
    return records


def _check_binary_dataset() -> list[dict]:
    data = REPO / "data" / "wits_eval_cases_binary.jsonl"
    valid = {"auto_approve", "block"}
    records = []
    for line in data.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        assert "command" in r and "binary_verdict" in r and "shell" in r, r
        assert r["binary_verdict"] in valid, r
        assert r["shell"] in ("bash", "powershell"), r
        records.append(r)
    print(f"\nbinary dataset OK: {len(records)} valid records")
    print("  by binary_verdict:", dict(Counter(r["binary_verdict"] for r in records)))
    print("  by shell  :", dict(Counter(r["shell"] for r in records)))
    return records


def _split_check(records: list[dict], label_key: str, label_names: list[str]) -> None:
    SEED = 42
    label2id = {n: i for i, n in enumerate(label_names)}
    TEST_FRAC = 0.25
    by_class: dict[int, list] = {i: [] for i in range(len(label_names))}
    for r in records:
        by_class[label2id[r[label_key]]].append(r)
    rng = random.Random(SEED)
    for k in by_class:
        rng.shuffle(by_class[k])
    train_recs, test_recs = [], []
    for k, lst in by_class.items():
        n_te = max(1, int(round(len(lst) * TEST_FRAC))) if len(lst) >= 2 else 0
        test_recs.extend(lst[:n_te])
        train_recs.extend(lst[n_te:])
    train_d = Counter(r[label_key] for r in train_recs)
    test_d = Counter(r[label_key] for r in test_recs)
    print(f"  split: train n={len(train_recs)} dist={dict(train_d)}")
    print(f"         test  n={len(test_recs)}  dist={dict(test_d)}")


def main() -> int:
    _check_notebook(REPO / "wits_main.ipynb")
    _check_notebook(REPO / "wits_binary_main.ipynb")
    _check_notebook(REPO / "wits_transcript_main.ipynb")
    print("all code cells in both notebooks parse as valid Python")

    recs4 = _check_4class_dataset()
    _split_check(recs4, "verdict",
                 ["safe", "maybe_safe", "unsafe", "extremely_unsafe"])

    recs2 = _check_binary_dataset()
    _split_check(recs2, "binary_verdict", ["auto_approve", "block"])

    # D3 transcript dataset (binary, no verdict field).
    d3_path = REPO / "data" / "d3_transcript_cases.jsonl"
    if d3_path.exists():
        d3 = [json.loads(l) for l in d3_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        bad = [r for r in d3 if r.get("decision") not in ("allow", "block")]
        if bad:
            raise SystemExit(f"D3 has {len(bad)} rows with non-binary decision")
        from collections import Counter
        print(f"\nD3 transcript dataset OK: {len(d3)} valid records")
        print("  by decision :", dict(Counter(r["decision"] for r in d3)))
        print("  by split    :", dict(Counter(r["split"] for r in d3)))
        print("  by shell    :", dict(Counter(r["shell"] for r in d3)))
    else:
        print(f"\n(skipping D3 check — {d3_path.name} not generated yet)")

    print("\nSmoke test OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
