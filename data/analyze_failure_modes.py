"""Failure-mode analysis: why is C (WITS+judge) beating E (GNN+judge)?

Loads everything the pipeline_eval produced and slices the failures so
we can see, concretely:

  1. Which rows did C get right that E got wrong? (E's misses)
  2. Of E's misses, what's the GNN doing? Confident-wrong, or routed-
     to-judge-and-judge-was-wrong?
  3. Where in the GNN's confidence distribution do the failures cluster?
  4. Per-bucket breakdown of E's misses.
  5. For each E-miss, what does the WITS verdict + transcript look like?
"""
from __future__ import annotations

import hashlib
import json
import pickle
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from data.judge_runner import (  # noqa: E402
    JUDGE_SYSTEM, build_user_prompt, _cache_key, _CACHE,
)
from models.gnn.graph_classifier import GraphClassifier  # noqa: E402

DATA_DIR = REPO / "outputs"
DEVICE = torch.device("cpu")
LABEL_NAMES = ["allow", "block"]
LABEL2ID = {n: i for i, n in enumerate(LABEL_NAMES)}
ID2LABEL = {i: n for n, i in LABEL2ID.items()}
NUM_CLASSES = 2
TAU = 0.75


def find_tag() -> str:
    cs = sorted(DATA_DIR.glob("test_graphs_d3_transcript_*.pkl"))
    if not cs:
        raise SystemExit("no test cache")
    return cs[-1].stem.replace("test_graphs_", "")


def main() -> int:
    TAG = find_tag()
    print(f"TAG: {TAG}\n")

    DATASET = REPO / "data" / "d3_transcript_cases.jsonl"
    rows = [json.loads(l) for l in DATASET.read_text(encoding="utf-8").splitlines() if l.strip()]
    test_recs = [r for r in rows if r.get("split") == "test"]
    rec_by_name = {r["case_name"]: r for r in test_recs}

    with open(DATA_DIR / f"test_graphs_{TAG}.pkl", "rb") as f: test_graphs = pickle.load(f)
    with open(DATA_DIR / f"test_meta_{TAG}.pkl",   "rb") as f: test_meta    = pickle.load(f)

    # GNN inference.
    weighted = DATA_DIR / f"gnn_weighted_{TAG}"
    with open(weighted / "model_metadata.json") as f: mw = json.load(f)
    gnn = GraphClassifier(hidden_channel_dimensions=mw["hidden_channel_dimensions"],
                          num_classes=mw["num_classes"]).to(DEVICE)
    gnn.load_state_dict(torch.load(weighted / "model.pt", map_location=DEVICE))
    gnn.eval()
    gnn_pred, gnn_prob = [], []
    loader = DataLoader(test_graphs, batch_size=1, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            logits = gnn(batch.x.float(), batch.edge_index, batch.batch, dropout_percentage=0.0)
            prob = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            gnn_pred.append(int(prob.argmax()))
            gnn_prob.append(prob)
    gnn_pred = np.asarray(gnn_pred)
    gnn_prob = np.asarray(gnn_prob)

    # WITS shim.
    SHIM_IN = DATA_DIR / f"d3_wits_input_{TAG}.jsonl"
    if not SHIM_IN.exists():
        with open(SHIM_IN, "w", encoding="utf-8", newline="\n") as f:
            for m in test_meta:
                rec = rec_by_name[m["case_name"]]
                f.write(json.dumps({"command": rec["proposed_command"], "shell": m["shell"]}, ensure_ascii=False) + "\n")
    import os as _os
    env = _os.environ.copy()
    env["WITS_DIST"] = _os.environ.get("WITS_DIST", "c:/dev/what-in-the-shell-fresh/dist/index.cjs")
    proc = subprocess.run(
        ["node", str(REPO / "data" / "_wits_score_shim.cjs")],
        input=SHIM_IN.read_text(encoding="utf-8"),
        capture_output=True, text=True, env=env, check=False,
        encoding="utf-8", errors="replace",
    )
    wits_preds = [json.loads(l) for l in proc.stdout.splitlines() if l.strip()]

    # Build the table.
    rows_table = []
    for i, m in enumerate(test_meta):
        rec = rec_by_name[m["case_name"]]
        w = wits_preds[i]
        # Compute C / E decisions exactly as run_pipeline_eval does.
        wv = w["verdict"]
        gnn_p = ID2LABEL[int(gnn_pred[i])]
        gnn_c = float(gnn_prob[i].max())
        # Look up the cached judge response (if any).
        analysis = {
            "rule_hits": [{"ruleId": rid, "severity": "info", "message": ""} for rid in w.get("rule_ids", [])],
            "effects":   [],
        }
        user_p = build_user_prompt(
            command=rec["proposed_command"], intention=rec.get("intention",""),
            transcript=rec["transcript"], analysis=analysis,
        )
        key = _cache_key("claude-sonnet-4.6", JUDGE_SYSTEM, user_p)
        judge_entry = _CACHE.get(key)
        judge_text = judge_entry.text if judge_entry else ""
        judge_decision_raw = None
        try:
            from data.judge_runner import parse_judge_response
            jd, _ = parse_judge_response(judge_text)
            judge_decision_raw = jd
        except Exception:
            judge_decision_raw = None
        judge_empty = (judge_text.strip() == "")

        # C: WITS shortcircuit, else judge (fail-closed to block on parse fail).
        if wv == "safe":
            c_dec = "allow"; c_route = "wits-shortcircuit-safe"
        elif wv == "extremely_unsafe":
            c_dec = "block"; c_route = "wits-shortcircuit-extreme"
        else:
            if judge_decision_raw == "auto_approve":
                c_dec = "allow"; c_route = "judge-allow"
            elif judge_decision_raw == "block":
                c_dec = "block"; c_route = "judge-block"
            else:
                c_dec = "block"; c_route = "judge-FAILED-fail-closed"
        # E: GNN-confident, else judge.
        if gnn_c >= TAU:
            e_dec = gnn_p; e_route = f"gnn-{gnn_p}-conf={gnn_c:.2f}"
        else:
            if judge_decision_raw == "auto_approve":
                e_dec = "allow"; e_route = "judge-allow"
            elif judge_decision_raw == "block":
                e_dec = "block"; e_route = "judge-block"
            else:
                e_dec = "block"; e_route = "judge-FAILED-fail-closed"

        rows_table.append({
            "case_name":     m["case_name"],
            "truth":         rec["decision"],
            "bucket":        rec.get("report_bucket", ""),
            "wits_verdict":  wv,
            "gnn_pred":      gnn_p,
            "gnn_conf":      round(gnn_c, 3),
            "judge_decision":judge_decision_raw or "(unparseable)",
            "judge_empty":   judge_empty,
            "C_dec":         c_dec,
            "C_route":       c_route,
            "C_right":       c_dec == rec["decision"],
            "E_dec":         e_dec,
            "E_route":       e_route,
            "E_right":       e_dec == rec["decision"],
            "command_short": rec["proposed_command"][:80],
        })
    df = pd.DataFrame(rows_table)

    print("=" * 100)
    print(f"OVERALL: C right = {df['C_right'].sum()}/{len(df)}, "
          f"E right = {df['E_right'].sum()}/{len(df)}")
    print("=" * 100)

    # Empty/parse-failed judge call counts.
    n_judge_called_for_c = int(df["wits_verdict"].isin(["maybe_safe","unsafe"]).sum())
    n_judge_called_for_e = int((df["gnn_conf"] < TAU).sum())
    n_judge_empty_c = int(df[df["wits_verdict"].isin(["maybe_safe","unsafe"])]["judge_empty"].sum())
    n_judge_empty_e = int(df[df["gnn_conf"] < TAU]["judge_empty"].sum())
    n_judge_parse_fail_c = int(df[df["wits_verdict"].isin(["maybe_safe","unsafe"])]["judge_decision"].eq("(unparseable)").sum())
    n_judge_parse_fail_e = int(df[df["gnn_conf"] < TAU]["judge_decision"].eq("(unparseable)").sum())
    print(f"\nJudge invocations: C={n_judge_called_for_c}  E={n_judge_called_for_e}")
    print(f"  C: of {n_judge_called_for_c}, judge returned empty in {n_judge_empty_c} cases "
          f"(fail-closed to BLOCK)")
    print(f"  E: of {n_judge_called_for_e}, judge returned empty in {n_judge_empty_e} cases "
          f"(fail-closed to BLOCK)")
    print(f"  C unparseable (incl. empty): {n_judge_parse_fail_c}")
    print(f"  E unparseable (incl. empty): {n_judge_parse_fail_e}")

    # Whether the fail-closed defaults happen to be CORRECT.
    fc_c = df[(df["wits_verdict"].isin(["maybe_safe","unsafe"])) & (df["judge_decision"] == "(unparseable)")]
    fc_e = df[(df["gnn_conf"] < TAU) & (df["judge_decision"] == "(unparseable)")]
    print(f"\n  Of C's fail-closed rows, truth distribution: {dict(Counter(fc_c['truth']))}")
    print(f"    'block' truth ones happened to land right; 'allow' ones became false positives.")
    print(f"  Of E's fail-closed rows, truth distribution: {dict(Counter(fc_e['truth']))}")

    # E-misses where C got it right.
    e_misses_c_right = df[(df["E_right"] == False) & (df["C_right"] == True)]
    print(f"\n{'=' * 100}")
    print(f"E-MISSES WHERE C GOT IT RIGHT ({len(e_misses_c_right)} rows)")
    print(f"{'=' * 100}")
    print(e_misses_c_right[
        ["case_name","truth","bucket","wits_verdict","gnn_pred","gnn_conf","C_route","E_route","command_short"]
    ].to_string(index=False))

    # E-misses by GNN confidence band.
    print(f"\n{'-' * 100}")
    print("E-misses by GNN-confidence band (was the model confident-wrong, or did the judge fail?):")
    print(f"{'-' * 100}")
    em = df[~df["E_right"]].copy()
    def _band(c):
        if c >= 0.9: return ">=0.90"
        if c >= 0.8: return "0.80-0.89"
        if c >= 0.7: return "0.70-0.79"
        if c >= 0.6: return "0.60-0.69"
        return "<0.60"
    em["band"] = em["gnn_conf"].apply(_band)
    em["route_class"] = em["E_route"].str.split("-").str[0]
    pivot = em.pivot_table(index="band", columns="route_class", values="case_name",
                            aggfunc="count", fill_value=0)
    print(pivot.to_string())

    # The biggest insight: when GNN sends to judge (low confidence), does the
    # judge save it?
    print(f"\n{'-' * 100}")
    print("Of E's judge-routed rows (gnn_conf < tau), how did the judge decide?")
    print(f"{'-' * 100}")
    erouted = df[df["gnn_conf"] < TAU]
    print(f"  total judge-routed: {len(erouted)}")
    print(f"  judge said allow:   {(erouted['judge_decision']=='auto_approve').sum()}")
    print(f"  judge said block:   {(erouted['judge_decision']=='block').sum()}")
    print(f"  judge empty/fail:   {(erouted['judge_decision']=='(unparseable)').sum()}  (forced to BLOCK)")
    print(f"  of judge-routed correctness: {(erouted['E_right']).sum()}/{len(erouted)}")

    # And vice versa for C.
    print(f"\n{'-' * 100}")
    print("Of C's judge-routed rows, how did the judge decide?")
    print(f"{'-' * 100}")
    crouted = df[df["wits_verdict"].isin(["maybe_safe","unsafe"])]
    print(f"  total judge-routed: {len(crouted)}")
    print(f"  judge said allow:   {(crouted['judge_decision']=='auto_approve').sum()}")
    print(f"  judge said block:   {(crouted['judge_decision']=='block').sum()}")
    print(f"  judge empty/fail:   {(crouted['judge_decision']=='(unparseable)').sum()}  (forced to BLOCK)")
    print(f"  of judge-routed correctness: {(crouted['C_right']).sum()}/{len(crouted)}")

    # GNN confident-wrong (where the GNN BYPASSED the judge AND was wrong).
    confwrong = df[(df["gnn_conf"] >= TAU) & (~df["E_right"])]
    print(f"\n{'-' * 100}")
    print(f"GNN CONFIDENT-WRONG ({len(confwrong)} rows: GNN was sure, sidestepped judge, wrong):")
    print(f"{'-' * 100}")
    print(confwrong[
        ["case_name","truth","bucket","gnn_pred","gnn_conf","wits_verdict","command_short"]
    ].to_string(index=False))

    # Per-bucket E vs C.
    print(f"\n{'-' * 100}")
    print("Per-bucket: did E catch up to C anywhere?")
    print(f"{'-' * 100}")
    per_bucket = df.groupby("bucket").agg(
        n=("truth", "size"),
        C_acc=("C_right", "mean"),
        E_acc=("E_right", "mean"),
    )
    per_bucket["delta_E_minus_C"] = per_bucket["E_acc"] - per_bucket["C_acc"]
    print(per_bucket.round(3).sort_values("n", ascending=False).to_string())

    # Save the full table for reference.
    out = DATA_DIR / "failure_analysis.csv"
    df.to_csv(out, index=False)
    print(f"\nFull per-row table -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
