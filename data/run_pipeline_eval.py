"""Run pipeline_eval_main.ipynb's logic headlessly from the CLI.

Same 5-pipeline comparison the notebook does, but as a standalone
script so it can run in the venv without a Jupyter kernel.

Outputs are printed to stdout and also saved to
outputs/pipeline_eval_summary.json for downstream use.

Usage:
    $env:GITHUB_TOKEN = (gh auth token)
    & "C:\\...\\venv\\python.exe" data\\run_pipeline_eval.py
"""
from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score,
)
from torch_geometric.loader import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from data.judge_runner import JudgeRunner  # noqa: E402
from models.gnn.graph_classifier import GraphClassifier  # noqa: E402

DATA_DIR = REPO / "outputs"
DEVICE = torch.device("cpu")
LABEL_NAMES = ["allow", "block"]
LABEL2ID = {n: i for i, n in enumerate(LABEL_NAMES)}
ID2LABEL = {i: n for n, i in LABEL2ID.items()}
NUM_CLASSES = 2

TAU = 0.75
JUDGE_CONCURRENCY = 4
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "claude-sonnet-4.6")


def find_cache_tag() -> str:
    cands = sorted(DATA_DIR.glob("test_graphs_d3_transcript_*.pkl"))
    if not cands:
        raise SystemExit("no test_graphs_d3_transcript_*.pkl found in outputs/")
    return cands[-1].stem.replace("test_graphs_", "")


def main() -> int:
    TAG = find_cache_tag()
    print(f"Cache TAG: {TAG}\n")

    # ---- 1. Load D3 test split + meta ----
    DATASET = REPO / "data" / "d3_transcript_cases.jsonl"
    rows = [json.loads(l) for l in DATASET.read_text(encoding="utf-8").splitlines() if l.strip()]
    test_recs = [r for r in rows if r.get("split") == "test"]
    rec_by_name = {r["case_name"]: r for r in test_recs}
    print(f"D3 test split: {len(test_recs)} rows ({Counter(r['decision'] for r in test_recs)})")

    with open(DATA_DIR / f"test_graphs_{TAG}.pkl", "rb") as f: test_graphs = pickle.load(f)
    with open(DATA_DIR / f"test_meta_{TAG}.pkl",   "rb") as f: test_meta    = pickle.load(f)
    with open(DATA_DIR / f"test_extras_{TAG}.pkl", "rb") as f: test_extras  = pickle.load(f)
    print(f"  test_graphs={len(test_graphs)}  test_meta={len(test_meta)}")

    # ---- 2. Load weighted GNN (the §7b retrained one) ----
    weighted_dir = DATA_DIR / f"gnn_weighted_{TAG}"
    fallback_dir = DATA_DIR / f"gnn_model_{TAG}"
    gnn_dir = weighted_dir if weighted_dir.exists() else fallback_dir
    with open(gnn_dir / "model_metadata.json") as f:
        md = json.load(f)
    print(f"loaded GNN from {gnn_dir.name} (best_f1={md.get('best_macro_f1','?')})")

    gnn = GraphClassifier(
        hidden_channel_dimensions=md["hidden_channel_dimensions"],
        num_classes=md["num_classes"],
    ).to(DEVICE)
    gnn.load_state_dict(torch.load(gnn_dir / "model.pt", map_location=DEVICE))
    gnn.eval()

    # ---- 3. WITS shim over the D3 test rows ----
    SHIM_IN = DATA_DIR / f"d3_wits_input_{TAG}.jsonl"
    with open(SHIM_IN, "w", encoding="utf-8", newline="\n") as f:
        for m in test_meta:
            rec = rec_by_name[m["case_name"]]
            f.write(json.dumps({"command": rec["proposed_command"], "shell": m["shell"]}, ensure_ascii=False) + "\n")
    shim_path = REPO / "data" / "_wits_score_shim.cjs"
    env = os.environ.copy()
    env["WITS_DIST"] = os.environ.get("WITS_DIST", "c:/dev/what-in-the-shell-fresh/dist/index.cjs")
    print(f"\nRunning WITS shim on {len(test_meta)} D3 rows ...")
    proc = subprocess.run(
        ["node", str(shim_path)],
        input=SHIM_IN.read_text(encoding="utf-8"),
        capture_output=True, text=True, env=env, check=False,
        encoding="utf-8", errors="replace",
    )
    if proc.returncode != 0:
        print("STDERR:", proc.stderr[-1500:])
        raise RuntimeError(f"WITS shim exited {proc.returncode}")
    wits_preds = [json.loads(l) for l in proc.stdout.splitlines() if l.strip()]
    assert len(wits_preds) == len(test_meta)
    print(f"  WITS verdict dist: {dict(Counter(p['verdict'] for p in wits_preds))}")

    # ---- 4. GNN per-sample inference + latency ----
    loader = DataLoader(test_graphs, batch_size=1, shuffle=False)
    # warmup
    with torch.no_grad():
        for _ in range(min(5, len(test_graphs))):
            b = next(iter(loader))
            gnn(b.x.float(), b.edge_index, b.batch, dropout_percentage=0.0)
    gnn_pred, gnn_prob, gnn_lat = [], [], []
    with torch.no_grad():
        for batch in loader:
            t0 = time.perf_counter()
            logits = gnn(batch.x.float(), batch.edge_index, batch.batch, dropout_percentage=0.0)
            prob = torch.softmax(logits, dim=-1)
            gnn_lat.append((time.perf_counter() - t0) * 1000.0)
            gnn_pred.append(int(prob.argmax(dim=-1).item()))
            gnn_prob.append(prob.cpu().numpy()[0])
    gnn_pred = np.asarray(gnn_pred)
    gnn_prob = np.asarray(gnn_prob)
    gnn_lat  = np.asarray(gnn_lat)
    print(f"\nGNN inference: mean={gnn_lat.mean():.2f}ms p95={np.percentile(gnn_lat,95):.2f}ms")
    print(f"  GNN pred class balance: {Counter(ID2LABEL[int(p)] for p in gnn_pred)}")
    print(f"  GNN argmax accuracy alone: {(gnn_pred == np.asarray([LABEL2ID[m['decision']] for m in test_meta])).mean():.3f}")

    # ---- 5. Build contexts ----
    contexts = []
    for i, m in enumerate(test_meta):
        rec = rec_by_name[m["case_name"]]
        w = wits_preds[i]
        contexts.append({
            "i":            i,
            "case_name":    m["case_name"],
            "transcript":   rec["transcript"],
            "command":      rec["proposed_command"],
            "intention":    rec.get("intention", ""),
            "shell":        rec["shell"],
            "report_bucket":rec.get("report_bucket", ""),
            "truth":        rec["decision"],
            "wits_verdict": w["verdict"],
            "wits_rules":   w.get("rule_ids", []),
            "wits_lat_ms":  float(w.get("elapsed_ms") or 0.0),
            "gnn_pred":     ID2LABEL[int(gnn_pred[i])],
            "gnn_conf":     float(gnn_prob[i].max()),
            "gnn_lat_ms":   float(gnn_lat[i]),
        })

    # ---- 6. Identify rows needing the judge ----
    JUDGE_VERDICTS = {"maybe_safe", "unsafe"}
    wits_judge_idx = [c["i"] for c in contexts if c["wits_verdict"] in JUDGE_VERDICTS]
    gnn_judge_idx  = [c["i"] for c in contexts if c["gnn_conf"] < TAU]
    union_idx      = sorted(set(wits_judge_idx) | set(gnn_judge_idx))
    print(f"\nJudge invocations:")
    print(f"  C (WITS+judge): {len(wits_judge_idx)}/{len(contexts)}")
    print(f"  E (GNN+judge @ τ={TAU}): {len(gnn_judge_idx)}/{len(contexts)}")
    print(f"  unique rows hitting Sonnet: {len(union_idx)}")

    # ---- 7. Call Sonnet (with concurrency + cache) ----
    judge = JudgeRunner(model=JUDGE_MODEL)
    def _judge_one(c):
        analysis = {
            "rule_hits": [{"ruleId": rid, "severity": "info", "message": ""} for rid in c["wits_rules"]],
            "effects":   [],
        }
        res = judge.judge(
            command=c["command"], intention=c["intention"],
            transcript=c["transcript"], analysis=analysis,
        )
        return c["i"], res

    needed = [contexts[i] for i in union_idx]
    print(f"\nCalling Sonnet 4.6 on {len(needed)} rows (concurrency {JUDGE_CONCURRENCY}) ...")
    judge_results = {}
    t_judge_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=JUDGE_CONCURRENCY) as ex:
        futures = {ex.submit(_judge_one, c): c["i"] for c in needed}
        for n_done, fut in enumerate(as_completed(futures), 1):
            i, res = fut.result()
            judge_results[i] = res
            if n_done % 10 == 0 or n_done == len(needed):
                cached = sum(1 for r in judge_results.values() if r.cached)
                print(f"  {n_done}/{len(needed)} done ({cached} cached)")
    t_judge = time.perf_counter() - t_judge_start
    n_cached = sum(1 for r in judge_results.values() if r.cached)
    n_fresh  = len(judge_results) - n_cached
    n_parse_err = sum(1 for r in judge_results.values() if r.parse_error)
    print(f"\nJudge: {len(judge_results)} done in {t_judge:.1f}s "
          f"({n_cached} cached, {n_fresh} fresh, {n_parse_err} parse errors)")

    # ---- 8. Assemble per-pipeline decisions ----
    def _wits_strict(v):     return "allow" if v == "safe" else "block"
    def _wits_permissive(v):
        if v == "safe": return "allow"
        if v == "extremely_unsafe": return "block"
        return "allow"

    rows = []
    for c in contexts:
        i = c["i"]
        a_dec, a_lat = _wits_strict(c["wits_verdict"]), c["wits_lat_ms"]
        b_dec, b_lat = _wits_permissive(c["wits_verdict"]), c["wits_lat_ms"]
        # C: WITS + judge
        if c["wits_verdict"] in {"safe", "extremely_unsafe"}:
            c_dec, c_lat = _wits_strict(c["wits_verdict"]), c["wits_lat_ms"]
        else:
            jr = judge_results[i]
            c_dec = "allow" if jr.decision == "auto_approve" else "block"
            c_lat = c["wits_lat_ms"] + jr.latency_ms
        # D: GNN alone
        d_dec, d_lat = c["gnn_pred"], c["gnn_lat_ms"]
        # E: GNN + judge @ tau
        if c["gnn_conf"] >= TAU:
            e_dec, e_lat = c["gnn_pred"], c["gnn_lat_ms"]
        else:
            jr = judge_results[i]
            e_dec = "allow" if jr.decision == "auto_approve" else "block"
            e_lat = c["gnn_lat_ms"] + jr.latency_ms
        rows.append({
            "i": i, "case_name": c["case_name"], "truth": c["truth"],
            "wits_verdict": c["wits_verdict"], "gnn_pred": c["gnn_pred"], "gnn_conf": c["gnn_conf"],
            "A_dec": a_dec, "A_lat_ms": a_lat,
            "B_dec": b_dec, "B_lat_ms": b_lat,
            "C_dec": c_dec, "C_lat_ms": c_lat,
            "D_dec": d_dec, "D_lat_ms": d_lat,
            "E_dec": e_dec, "E_lat_ms": e_lat,
            "report_bucket": c["report_bucket"],
        })
    df = pd.DataFrame(rows)

    # ---- 9. Headline table ----
    PIPELINES = [
        ("A — WITS strict",          "A_dec", "A_lat_ms"),
        ("B — WITS permissive",      "B_dec", "B_lat_ms"),
        ("C — WITS + Sonnet judge",  "C_dec", "C_lat_ms"),
        ("D — GNN alone",            "D_dec", "D_lat_ms"),
        (f"E — GNN + judge @ τ={TAU}", "E_dec", "E_lat_ms"),
    ]
    yt = df["truth"].values
    summary = []
    for name, dcol, lcol in PIPELINES:
        yp = df[dcol].values
        invocations = (
            0 if dcol in ("A_dec", "B_dec", "D_dec")
            else int(df["wits_verdict"].isin(["maybe_safe", "unsafe"]).sum()) if dcol == "C_dec"
            else int((df["gnn_conf"] < TAU).sum())
        )
        lat = df[lcol].values
        summary.append({
            "pipeline":           name,
            "accuracy":           round(accuracy_score(yt, yp), 4),
            "macro_f1":           round(f1_score(yt, yp, average="macro", labels=LABEL_NAMES, zero_division=0), 4),
            "prec_block":         round(precision_score(yt, yp, pos_label="block", zero_division=0), 4),
            "recall_block":       round(recall_score(yt, yp, pos_label="block", zero_division=0), 4),
            "judge_invocations":  invocations,
            "judge_rate":         round(invocations / len(yt), 4),
            "lat_mean_ms":        round(float(lat.mean()), 2),
            "lat_p95_ms":         round(float(np.percentile(lat, 95)), 2),
            "lat_max_ms":         round(float(lat.max()), 2),
        })
    summary_df = pd.DataFrame(summary)

    print("\n" + "="*100)
    print("HEADLINE — 5 pipelines on the D3 test split (n={})".format(len(df)))
    print("="*100)
    print(summary_df.to_string(index=False))

    # ---- 10. Confusion matrices ----
    print("\n" + "-"*100)
    print("Confusion matrices (rows=true, cols=pred):")
    print("-"*100)
    for name, dcol, _ in PIPELINES:
        cm = confusion_matrix(df["truth"], df[dcol], labels=LABEL_NAMES)
        print(f"\n{name}:")
        print(pd.DataFrame(cm, index=LABEL_NAMES, columns=LABEL_NAMES).to_string())

    # ---- 11. C vs E head-to-head ----
    n = len(df)
    c_correct = int((df["C_dec"] == df["truth"]).sum())
    e_correct = int((df["E_dec"] == df["truth"]).sum())
    diff = df[df["C_dec"] != df["E_dec"]].copy()
    diff["C_right"] = diff["C_dec"] == diff["truth"]
    diff["E_right"] = diff["E_dec"] == diff["truth"]
    n_e_better = int((diff["E_right"] & ~diff["C_right"]).sum())
    n_c_better = int((diff["C_right"] & ~diff["E_right"]).sum())
    n_both_wr  = int((~diff["C_right"] & ~diff["E_right"]).sum())

    print("\n" + "="*100)
    print("THE PRODUCTION REPLACEMENT QUESTION — C (WITS+judge) vs E (GNN+judge)")
    print("="*100)
    print(f"  C: WITS+judge         acc={c_correct/n:.3f}  ({c_correct}/{n})  "
          f"mean_lat={df['C_lat_ms'].mean():.0f}ms  p95={np.percentile(df['C_lat_ms'],95):.0f}ms  "
          f"judge_calls={summary[2]['judge_invocations']}")
    print(f"  E: GNN+judge (τ={TAU})  acc={e_correct/n:.3f}  ({e_correct}/{n})  "
          f"mean_lat={df['E_lat_ms'].mean():.0f}ms  p95={np.percentile(df['E_lat_ms'],95):.0f}ms  "
          f"judge_calls={summary[4]['judge_invocations']}")
    print(f"\n  Disagreements C vs E: {len(diff)}")
    print(f"    E right, C wrong : {n_e_better}")
    print(f"    C right, E wrong : {n_c_better}")
    print(f"    both wrong (diff): {n_both_wr}")
    print(f"    net advantage to E: {n_e_better - n_c_better:+d}")

    # ---- 12. Tau sweep for E (no extra judge calls — all cached) ----
    print("\n" + "-"*100)
    print(f"Pipeline E tau sweep (same GNN, all judge calls cached):")
    print("-"*100)
    TAUS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
    sweep = []
    for tau in TAUS:
        invocations = 0
        preds = []
        for c in contexts:
            if c["gnn_conf"] >= tau:
                preds.append(c["gnn_pred"])
            else:
                jr = judge_results.get(c["i"])
                preds.append("allow" if jr and jr.decision == "auto_approve" else "block")
                invocations += 1
        sweep.append({
            "tau":          tau,
            "judge_rate":   round(invocations / len(contexts), 3),
            "accuracy":     round(accuracy_score(df["truth"], preds), 3),
            "macro_f1":     round(f1_score(df["truth"], preds, average="macro", labels=LABEL_NAMES, zero_division=0), 3),
            "recall_block": round(recall_score(df["truth"], preds, pos_label="block", zero_division=0), 3),
        })
    print(pd.DataFrame(sweep).to_string(index=False))

    # ---- 13. Per-bucket accuracy ----
    print("\n" + "-"*100)
    print("Per-bucket accuracy:")
    print("-"*100)
    for name, dcol, _ in PIPELINES:
        df[f"{dcol}_right"] = df[dcol] == df["truth"]
    by_bucket = df.groupby("report_bucket").agg(
        n=("truth", "size"),
        A=("A_dec_right", "mean"),
        B=("B_dec_right", "mean"),
        C=("C_dec_right", "mean"),
        D=("D_dec_right", "mean"),
        E=("E_dec_right", "mean"),
    ).sort_values("n", ascending=False)
    print(by_bucket.round(3).to_string())

    # ---- 14. Save summary JSON ----
    out_summary = DATA_DIR / "pipeline_eval_summary.json"
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump({
            "tag": TAG,
            "n_test": len(df),
            "tau": TAU,
            "judge_model": JUDGE_MODEL,
            "headline": summary,
            "tau_sweep": sweep,
            "C_vs_E": {
                "C_accuracy": c_correct/n,
                "E_accuracy": e_correct/n,
                "disagreements": len(diff),
                "E_right_C_wrong": n_e_better,
                "C_right_E_wrong": n_c_better,
                "both_wrong_diff": n_both_wr,
            },
            "judge_calls": {
                "total":   len(judge_results),
                "cached":  n_cached,
                "fresh":   n_fresh,
                "errors":  n_parse_err,
                "wall_seconds": round(t_judge, 1),
            },
        }, f, indent=2)
    print(f"\nSummary -> {out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
