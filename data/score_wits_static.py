"""Score the WITS static analyzer over our test split, head-to-head with the GNN.

Usage:
    python data/score_wits_static.py \
        --test-jsonl outputs/wits_test_split_<dataset_stem>.jsonl \
        --out outputs/wits_static_predictions.jsonl

Workflow:
    1. Read the held-out 4-class test split exported by wits_main.ipynb
       (section 2). Each row has the ground-truth WITS verdict we
       relabelled it with.
    2. Pipe the (command, shell) pairs through data/_wits_score_shim.cjs,
       which runs `whatInTheShell.isThis(...)` on each. WITS_DIST env var
       points at the WITS dist build (default: /c/dev/what-in-the-shell-fresh/dist/index.cjs).
    3. Join predictions back to ground truth and compute:
         - accuracy
         - per-class precision / recall / F1
         - 4x4 confusion matrix
         - per-source-bucket accuracy breakdown
         - failure-mode summary: cases where WITS short-circuits to `safe`
           on something we labelled unsafe/extreme (= silent auto-approve;
           the WITS review comment's exact complaint).
    4. Writes per-row predictions to --out (JSONL) and prints metrics.

The output JSONL is what the notebook (§11) reads to render the side-by-side
comparison with the GNN.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SHIM = REPO / "data" / "_wits_score_shim.cjs"
WITS_DIST_DEFAULT = "c:/dev/what-in-the-shell-fresh/dist/index.cjs"

VERDICTS = ["safe", "maybe_safe", "unsafe", "extremely_unsafe"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-jsonl", required=True,
                    help="Path to the exported test split (wits_test_split_*.jsonl).")
    ap.add_argument("--out", required=True,
                    help="Where to write WITS per-row predictions.")
    ap.add_argument("--wits-dist", default=WITS_DIST_DEFAULT,
                    help=f"Path to the WITS dist CJS bundle (default: {WITS_DIST_DEFAULT}).")
    ap.add_argument("--node", default="node",
                    help="Node executable (default: node).")
    return ap.parse_args()


def run_wits(records: list[dict], wits_dist: str, node: str) -> list[dict]:
    """Pipe records through the shim, return aligned predictions."""
    env = os.environ.copy()
    env["WITS_DIST"] = wits_dist
    print(f"Running WITS shim on {len(records)} records...")
    print(f"  node     = {node}")
    print(f"  shim     = {SHIM}")
    print(f"  wits dist= {wits_dist}")

    input_lines = "\n".join(
        json.dumps({"command": r["command"], "shell": r.get("shell", "bash")})
        for r in records
    ) + "\n"

    proc = subprocess.run(
        [node, str(SHIM)],
        input=input_lines,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        print("STDERR:", proc.stderr, file=sys.stderr)
        raise SystemExit(f"node shim exited {proc.returncode}")

    out_lines = [l for l in proc.stdout.splitlines() if l.strip()]
    if len(out_lines) != len(records):
        print(f"WARNING: expected {len(records)} predictions, got {len(out_lines)}.")
        print("STDERR:", proc.stderr, file=sys.stderr)

    return [json.loads(l) for l in out_lines]


def precision_recall_f1(y_true: list[str], y_pred: list[str], cls: str) -> tuple[float, float, float, int]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
    support = sum(1 for t in y_true if t == cls)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1, support


def report(y_true: list[str], y_pred: list[str], records: list[dict],
           latencies_ms: list[float] | None = None) -> None:
    n = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    print(f"\n{'='*70}")
    print(f"WITS static-analyzer 4-class metrics on {n} test cases")
    print(f"{'='*70}")
    print(f"  accuracy = {correct/n:.3f}   ({correct}/{n})")

    if latencies_ms is not None and latencies_ms:
        import statistics
        lats = sorted(latencies_ms)
        def pct(p): return lats[min(len(lats) - 1, int(round(p * len(lats))))]
        print(f"\n  per-call latency (ms): "
              f"mean={statistics.mean(lats):.2f}  median={statistics.median(lats):.2f}  "
              f"p95={pct(0.95):.2f}  p99={pct(0.99):.2f}  max={max(lats):.2f}")
        print(f"  total wall-clock (ms): {sum(lats):.0f}  "
              f"(amortized {sum(lats)/len(lats):.2f} ms/case across {len(lats)} cases)")

    print(f"\n  per-class precision / recall / F1:")
    print(f"  {'class':<20s} {'precision':>10s} {'recall':>10s} {'F1':>10s} {'support':>10s}")
    macro_p = macro_r = macro_f = 0.0
    macro_n = 0
    for cls in VERDICTS:
        p, r, f, s = precision_recall_f1(y_true, y_pred, cls)
        print(f"  {cls:<20s} {p:>10.3f} {r:>10.3f} {f:>10.3f} {s:>10d}")
        if s > 0:
            macro_p += p; macro_r += r; macro_f += f; macro_n += 1
    if macro_n:
        print(f"  {'(macro average)':<20s} {macro_p/macro_n:>10.3f} "
              f"{macro_r/macro_n:>10.3f} {macro_f/macro_n:>10.3f}")

    print(f"\n  confusion matrix (rows = true, cols = WITS prediction):")
    cm = {(t, p): 0 for t in VERDICTS for p in VERDICTS}
    for t, p in zip(y_true, y_pred):
        if p in VERDICTS:
            cm[(t, p)] += 1
    header = "  " + " " * 20 + " ".join(f"{p[:7]:>9s}" for p in VERDICTS) + "   total"
    print(header)
    for t in VERDICTS:
        row = " ".join(f"{cm[(t,p)]:>9d}" for p in VERDICTS)
        tot = sum(cm[(t,p)] for p in VERDICTS)
        print(f"  {t:<20s} {row}   {tot}")

    # Production-relevant failure surface (the review-comment's exact complaint).
    print(f"\n  Silent auto-approves (WITS says `safe` but truth is unsafe/extreme):")
    silent = [(r, t, p) for r, t, p in zip(records, y_true, y_pred)
              if p == "safe" and t in ("unsafe", "extremely_unsafe")]
    print(f"    {len(silent)} cases")
    for r, t, p in silent[:30]:
        src = r.get("source", "")[:25]
        print(f"      [{t:<16s}] {r['command'][:90]!r}   src={src}")
    if len(silent) > 30:
        print(f"      ... and {len(silent) - 30} more")

    print(f"\n  Silent hard-deny (WITS says `extremely_unsafe` but truth is safe/maybe):")
    deny = [(r, t, p) for r, t, p in zip(records, y_true, y_pred)
            if p == "extremely_unsafe" and t in ("safe", "maybe_safe")]
    print(f"    {len(deny)} cases")
    for r, t, p in deny[:15]:
        src = r.get("source", "")[:25]
        print(f"      [{t:<16s}] {r['command'][:90]!r}   src={src}")

    # Per-source bucket accuracy.
    by_src = defaultdict(lambda: [0, 0])
    for r, t, p in zip(records, y_true, y_pred):
        src = r.get("source", "(unknown)")
        bucket = src.split(":", 1)[0] if ":" in src else src
        by_src[bucket][0] += 1
        if t == p:
            by_src[bucket][1] += 1
    print(f"\n  Per-source-bucket accuracy:")
    print(f"  {'bucket':<24s} {'n':>5s} {'correct':>8s} {'acc':>6s}")
    for b, (n_, c) in sorted(by_src.items(), key=lambda kv: -kv[1][0]):
        print(f"  {b:<24s} {n_:>5d} {c:>8d} {c/n_:>6.1%}")


def main() -> int:
    args = parse_args()
    test_path = Path(args.test_jsonl)
    out_path = Path(args.out)
    if not test_path.exists():
        print(f"ERROR: test split not found at {test_path}", file=sys.stderr)
        print("       Run wits_main.ipynb section 2 (test-split export) first.", file=sys.stderr)
        return 2
    records = [json.loads(l) for l in test_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"Loaded {len(records)} test records from {test_path.name}")
    by_v = Counter(r.get("verdict", r.get("binary_verdict", "?")) for r in records)
    print(f"  by ground-truth verdict: {dict(by_v)}")

    preds = run_wits(records, args.wits_dist, args.node)

    # Join: same order assumed (shim preserves order).
    if len(preds) != len(records):
        print(f"ERROR: prediction count mismatch ({len(preds)} vs {len(records)}). Aborting.", file=sys.stderr)
        return 3

    joined = []
    for rec, pred in zip(records, preds):
        joined.append({
            **rec,
            "wits_verdict": pred.get("verdict"),
            "wits_rule_ids": pred.get("rule_ids", []),
            "wits_elapsed_ms": pred.get("elapsed_ms"),
            "wits_error": pred.get("error"),
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for r in joined:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(joined)} predictions -> {out_path}")

    y_true = [r["verdict"] for r in records]
    y_pred = [p.get("verdict") or "(error)" for p in preds]
    latencies = [float(p.get("elapsed_ms") or 0.0) for p in preds]

    n_errs = sum(1 for p in preds if p.get("error"))
    if n_errs:
        print(f"\nNOTE: {n_errs} WITS calls errored; they count as misclassifications.")

    report(y_true, y_pred, records, latencies_ms=latencies)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
