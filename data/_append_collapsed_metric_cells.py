"""Append the operational 3-class collapse cells (§11b) to wits_main.ipynb in-place.

Why a separate appender instead of regenerating from _build_wits_notebook.py:
the user's notebook is open in a live Jupyter kernel and may have outputs /
in-flight execution. Rebuilding from scratch would clobber that. This script
just APPENDS two cells at the end and rewrites the file with json indent=1
matching _build_wits_notebook.py's emitter.

Idempotent: re-running detects the marker comment in the first new cell and
no-ops if §11b is already present.

Run:
    python data/_append_collapsed_metric_cells.py
"""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

NB = Path(r"C:/dev/attention-graph-injection-detector/wits_main.ipynb")
MARKER = "# >>> SECTION 11b: operational 3-class collapse <<<"


def md(s: str) -> dict:
    s = dedent(s).strip("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}


def code(s: str) -> dict:
    s = dedent(s).strip("\n") + "\n"
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": s.splitlines(keepends=True),
    }


SECTION_MD = md("""
    ## 11b. Operational 3-class collapse (`maybe_safe ∪ unsafe → maybe`)

    In production WITS / our gate uses a **two-stage** pipeline:

    | static-analyzer verdict | what happens at runtime |
    | --- | --- |
    | `safe` | auto-approve (no LLM call) |
    | `extremely_unsafe` | hard-deny (no LLM call) |
    | `maybe_safe` *or* `unsafe` | **invoke the LLM judge** |

    So at deployment time, the distinction between `maybe_safe` and
    `unsafe` carries **zero information** — both deterministically route
    to the judge. Penalising the model for confusing them is a false-
    precision penalty.

    This section reports the **collapsed 3-class metric**
    (`safe / maybe / extremely_unsafe`) for every method, alongside the
    original 4-class numbers, so we can see how much the
    maybe/unsafe distinction is hurting each method's apparent score.

    Training keeps the full 4-class granularity (the finer signal helps
    the model learn the boundary even if production collapses it). Only
    the *evaluation* lens changes.
""")


COLLAPSE_CELL = code(f"""
    {MARKER}
    # Map 4-class verdicts to the 3-class operational view.
    def to_operational(v):
        if v in ("maybe_safe", "unsafe"):
            return "maybe"
        return v

    OP_LABELS = ["safe", "maybe", "extremely_unsafe"]

    def collapse_preds(y_true_4, y_pred_4):
        y_true_str = [v if isinstance(v, str) else ID2LABEL[int(v)] for v in y_true_4]
        y_pred_str = [v if isinstance(v, str) else ID2LABEL[int(v)] for v in y_pred_4]
        return [to_operational(v) for v in y_true_str], [to_operational(v) for v in y_pred_str]

    # Pull every method's (y_true, y_pred) into a uniform place. We use
    # the same predictions the §8/§11 cells produced — no re-running.
    method_runs = {{
        "WITS static (rule-based)":      (aligned_y_true, aligned_wits_pred),
        "Prompt-only LLM (1-token)":     (yp_test,        pp_test),
        "GNN (unweighted CE)":           (gnn_unweighted_eval["y_true"], gnn_unweighted_eval["y_pred"]),
        "GNN (class-weighted CE)":       (gnn_weighted_eval["y_true"],   gnn_weighted_eval["y_pred"]),
    }}

    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

    rows = []
    for name, (yt, yp) in method_runs.items():
        # 4-class as-is (convert ints to strings if needed)
        yt4 = [v if isinstance(v, str) else ID2LABEL[int(v)] for v in yt]
        yp4 = [v if isinstance(v, str) else ID2LABEL[int(v)] for v in yp]
        acc4 = accuracy_score(yt4, yp4)
        f14  = f1_score(yt4, yp4, average='macro', labels=LABEL_NAMES, zero_division=0)
        # 3-class operational collapse
        yt3, yp3 = collapse_preds(yt, yp)
        acc3 = accuracy_score(yt3, yp3)
        f13  = f1_score(yt3, yp3, average='macro', labels=OP_LABELS, zero_division=0)
        rows.append({{
            "method":             name,
            "acc_4cls":           acc4,
            "macro_f1_4cls":      f14,
            "acc_3cls_op":        acc3,
            "macro_f1_3cls_op":   f13,
            "delta_acc":          acc3 - acc4,
        }})
    op_summary = pd.DataFrame(rows)
    print("4-class vs operational 3-class (maybe_safe ∪ unsafe = `maybe`):")
    op_summary
""")


PER_METHOD_CELL = code("""
    # Per-method 3-class classification reports + confusion matrices.
    for name, (yt, yp) in method_runs.items():
        yt3, yp3 = collapse_preds(yt, yp)
        print("=" * 78)
        print(name + " — operational 3-class")
        print("=" * 78)
        print(f"  accuracy   = {accuracy_score(yt3, yp3):.3f}")
        print(f"  macro F1   = {f1_score(yt3, yp3, average='macro', labels=OP_LABELS, zero_division=0):.3f}")
        print(classification_report(yt3, yp3, labels=OP_LABELS,
                                    target_names=OP_LABELS, zero_division=0))
        cm = confusion_matrix(yt3, yp3, labels=OP_LABELS)
        print("Confusion (rows=true, cols=pred):")
        print(pd.DataFrame(cm, index=OP_LABELS, columns=OP_LABELS))
        print()
""")


JUDGE_LOAD_CELL = code("""
    # ---- Token / latency budget view: what fraction of commands can each
    # method short-circuit WITHOUT invoking the LLM judge?
    #
    # In the production pipeline, the judge is invoked iff the static
    # verdict is `maybe` (= maybe_safe ∪ unsafe). So:
    #
    #   short_circuit_rate = (#safe + #extremely_unsafe predicted) / n
    #
    # Higher is better (saves tokens + latency). But this only counts if
    # the short-circuit was CORRECT — silently auto-approving a
    # dangerous command or hard-denying a benign one is worse than
    # invoking the judge. So we report two columns:
    #
    #   short_circuit_rate         — % of commands NOT sent to judge
    #   safe_short_circuit_rate    — % of commands not sent to judge AND
    #                                whose prediction matched the truth
    #                                in the collapsed 3-class metric

    rows = []
    for name, (yt, yp) in method_runs.items():
        yt3, yp3 = collapse_preds(yt, yp)
        n = len(yt3)
        # Predictions that bypass the judge.
        bypass = [(t, p) for t, p in zip(yt3, yp3) if p != "maybe"]
        n_bypass = len(bypass)
        # Of those bypasses, how many were correct? An incorrect bypass
        # is a silent error: the agent acts/blocks without consulting
        # the judge when it should have.
        n_correct_bypass = sum(1 for t, p in bypass if t == p)
        # Of those bypasses, how many were SILENT-UNSAFE
        # (predicted `safe` but truth needed the judge or worse)?
        n_silent_unsafe = sum(1 for t, p in bypass
                              if p == "safe" and t in ("maybe", "extremely_unsafe"))
        n_silent_block = sum(1 for t, p in bypass
                             if p == "extremely_unsafe" and t in ("safe", "maybe"))
        rows.append({
            "method":                       name,
            "short_circuit_rate":           n_bypass / n,
            "correct_short_circuit_rate":   n_correct_bypass / n,
            "silent_auto_approve_rate":     n_silent_unsafe / n,
            "silent_hard_deny_rate":        n_silent_block / n,
            "judge_invocations":            n - n_bypass,
            "n_test":                       n,
        })
    budget_summary = pd.DataFrame(rows)
    print("Production token/latency-budget view:")
    print("  short_circuit_rate          = fraction of commands NOT sent to LLM judge")
    print("  correct_short_circuit_rate  = of those, fraction where the shortcut was right")
    print("  silent_auto_approve_rate    = predicted `safe` but truth was `maybe`/`extremely_unsafe`  ← THE failure mode")
    print("  silent_hard_deny_rate       = predicted `extremely_unsafe` but truth was `safe`/`maybe`")
    print()
    budget_summary
""")


def main() -> int:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    # Idempotency: bail if marker already in any code cell.
    for c in nb["cells"]:
        if c.get("cell_type") == "code" and MARKER in "".join(c.get("source", [])):
            print(f"§11b already present in {NB.name} — no-op.")
            return 0

    before = len(nb["cells"])
    nb["cells"].append(SECTION_MD)
    nb["cells"].append(COLLAPSE_CELL)
    nb["cells"].append(PER_METHOD_CELL)
    nb["cells"].append(JUDGE_LOAD_CELL)
    after = len(nb["cells"])

    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Appended §11b: {before} -> {after} cells in {NB.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
