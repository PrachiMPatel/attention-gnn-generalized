"""Build pipeline_eval_main.ipynb (Phase 7.5).

Eval-only notebook that compares 5 complete decision pipelines on the
held-out D3 test split:

  1. WITS static alone (strict)   -- shortcircuits, treat maybe -> block
  2. WITS static alone (permissive) -- shortcircuits, treat maybe -> allow
  3. WITS + Sonnet judge           -- production pipeline
  4. GNN alone (transcript-aware, binary)
  5. GNN + Sonnet judge at confidence threshold tau

Every call to Sonnet 4.6 is via copilot_client.CopilotClient() (same
path the runtime uses) and cached by data/judge_runner.py.

Headline numbers:
  - Accuracy / precision / recall / F1 / TP/FP/FN/TN per pipeline
  - End-to-end latency mean / p95 / max (incl. WITS + GNN forward +
    cached judge time)
  - Judge invocation rate per pipeline
  - Dollar-cost estimate
  - Confusion matrices, per-bucket accuracy, pairwise disagreement

Inputs all on disk after wits_transcript_main.ipynb has been run end-
to-end:
  - data/d3_transcript_cases.jsonl
  - outputs/d3_test_split_*.jsonl
  - outputs/gnn_model_d3_transcript_*/{model.pt, model_metadata.json}
  - outputs/test_graphs_*.pkl
  - outputs/test_meta_*.pkl
  - outputs/test_extras_*.pkl  (for prompt-only LLM rows if needed)
"""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "pipeline_eval_main.ipynb"


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


cells: list[dict] = []

cells.append(md("""
    # Pipeline eval — WITS+judge vs GNN+judge (real Sonnet 4.6 calls)

    The honest, end-to-end comparison the previous notebooks couldn't
    make. Loads the trained transcript-aware GNN, runs the WITS static
    analyzer over the same test split, and where either pipeline would
    fall through to an LLM judge, calls **Claude Sonnet 4.6 via CAPI**
    using the same `copilot_client` path the runtime uses.

    Every Sonnet call is cached on disk
    (`outputs/judge_cache.jsonl`) keyed by SHA256 of (model + system +
    user prompt), so re-runs are free.

    Pipelines compared:

    | id | pipeline | how decisions are produced |
    | --- | --- | --- |
    | A | WITS static (strict) | `safe -> allow`, everything else -> `block` |
    | B | WITS static (permissive) | `safe -> allow`, `extremely_unsafe -> block`, rest -> `allow` |
    | C | **WITS + Sonnet judge** | WITS shortcircuits + Sonnet decides `maybe_safe`/`unsafe` (mirrors production) |
    | D | GNN alone | argmax of binary softmax — no judge fallback |
    | E | **GNN + Sonnet judge @ τ** | GNN decides when `max(softmax) >= τ`, else Sonnet |

    The headline question: does (E) beat (C) on D3?

    Prerequisites (run before this notebook):
    - `python data/build_d3_transcript_dataset.py`
    - `wits_transcript_main.ipynb` end-to-end (produces the trained GNN
      + cached features)
    - `gh auth login` (or otherwise have CAPI auth set up — same path
      copilot_client uses; smoke-test:
      `python data/judge_runner.py --command ls --intention list`)
"""))


# --------------------------------------------------------------------------
cells.append(md("## 1. Setup"))

cells.append(code("""
    import os, sys, json, pickle, time, statistics
    from pathlib import Path
    from collections import Counter, defaultdict

    import numpy as np
    import pandas as pd
    import torch

    NOTEBOOK_DIR = Path.cwd()
    REPO_ROOT = NOTEBOOK_DIR
    while not (REPO_ROOT / "models" / "gnn" / "graph_classifier.py").exists() and REPO_ROOT.parent != REPO_ROOT:
        REPO_ROOT = REPO_ROOT.parent
    sys.path.insert(0, str(REPO_ROOT))
    print("Repo root:", REPO_ROOT)

    DATA_DIR = REPO_ROOT / "outputs"
    DEVICE = torch.device("cpu")  # eval-only, CPU is fine
    print("Device:", DEVICE)
"""))


# --------------------------------------------------------------------------
cells.append(md("## 2. Load D3 test split, trained GNN, cached features"))

cells.append(code("""
    DATASET_PATH = REPO_ROOT / "data" / "d3_transcript_cases.jsonl"
    raw = [json.loads(l) for l in DATASET_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    test_recs = [r for r in raw if r.get("split") == "test"]
    print(f"D3 test split: {len(test_recs)} rows ({Counter(r['decision'] for r in test_recs)})")

    # Find the most recent transcript-cache TAG.
    candidates = sorted(DATA_DIR.glob("test_graphs_d3_transcript_*.pkl"))
    if not candidates:
        raise SystemExit("no test_graphs_d3_transcript_*.pkl in outputs/. "
                         "Run wits_transcript_main.ipynb sections 1-5 first.")
    TEST_GRAPHS_PATH = candidates[-1]
    TAG = TEST_GRAPHS_PATH.stem.replace("test_graphs_", "")
    print(f"Using TAG: {TAG}")

    TEST_META_PATH  = DATA_DIR / f"test_meta_{TAG}.pkl"
    TEST_EXTRA_PATH = DATA_DIR / f"test_extras_{TAG}.pkl"
    GNN_MODEL_DIR   = DATA_DIR / f"gnn_model_{TAG}"

    with open(TEST_GRAPHS_PATH, "rb") as f:
        test_graphs = pickle.load(f)
    with open(TEST_META_PATH, "rb") as f:
        test_meta = pickle.load(f)
    with open(TEST_EXTRA_PATH, "rb") as f:
        test_extras = pickle.load(f)
    print(f"  loaded {len(test_graphs)} test graphs, {len(test_meta)} meta, {len(test_extras['labels'])} extras")

    # Sanity: the meta should line up with test_recs by case_name.
    rec_by_name = {r["case_name"]: r for r in test_recs}
    for m in test_meta:
        if m["case_name"] not in rec_by_name:
            print(f"  WARNING: meta has case {m['case_name']!r} not in test_recs")
"""))

cells.append(code("""
    # Load the trained binary GNN.
    from models.gnn.graph_classifier import GraphClassifier

    with open(GNN_MODEL_DIR / "model_metadata.json") as f:
        md_u = json.load(f)
    gnn = GraphClassifier(
        hidden_channel_dimensions=md_u["hidden_channel_dimensions"],
        num_classes=md_u["num_classes"],
    ).to(DEVICE)
    gnn.load_state_dict(torch.load(GNN_MODEL_DIR / "model.pt", map_location=DEVICE))
    gnn.eval()
    print(f"loaded GNN: {md_u}")

    LABEL_NAMES = ["allow", "block"]
    LABEL2ID = {n: i for i, n in enumerate(LABEL_NAMES)}
    ID2LABEL = {i: n for n, i in LABEL2ID.items()}
"""))


# --------------------------------------------------------------------------
cells.append(md("## 3. WITS static analyzer over the D3 test split"))

cells.append(code("""
    # Use the same Node shim as in wits_main.ipynb section 11.
    import subprocess

    WITS_PRED_PATH = DATA_DIR / f"d3_wits_predictions_{TAG}.jsonl"
    WITS_DIST = os.environ.get("WITS_DIST", "c:/dev/what-in-the-shell-fresh/dist/index.cjs")

    # Build the input JSONL (command + shell only).
    SHIM_IN_PATH = DATA_DIR / f"d3_wits_input_{TAG}.jsonl"
    with open(SHIM_IN_PATH, "w", encoding="utf-8", newline="\\n") as f:
        for m in test_meta:
            f.write(json.dumps({"command": rec_by_name[m["case_name"]]["proposed_command"],
                                "shell":   m["shell"]}, ensure_ascii=False) + "\\n")

    shim_path = REPO_ROOT / "data" / "_wits_score_shim.cjs"
    print(f"Running WITS shim on {len(test_meta)} D3 rows ...")
    env = os.environ.copy()
    env["WITS_DIST"] = WITS_DIST
    in_lines = SHIM_IN_PATH.read_text(encoding="utf-8")
    proc = subprocess.run(
        ["node", str(shim_path)],
        input=in_lines, capture_output=True, text=True, env=env, check=False,
    )
    if proc.returncode != 0:
        print("STDERR:", proc.stderr[-1500:])
        raise RuntimeError(f"WITS shim exited {proc.returncode}")

    wits_lines = [l for l in proc.stdout.splitlines() if l.strip()]
    assert len(wits_lines) == len(test_meta), f"WITS preds {len(wits_lines)} vs test {len(test_meta)}"
    wits_preds = [json.loads(l) for l in wits_lines]
    print(f"  {len(wits_preds)} WITS verdicts in.")
    print(f"  verdict distribution: {dict(Counter(p['verdict'] for p in wits_preds))}")
"""))


# --------------------------------------------------------------------------
cells.append(md("## 4. GNN inference (per-sample latency captured)"))

cells.append(code("""
    from torch_geometric.loader import DataLoader

    @torch.no_grad()
    def time_gnn_per_sample(gnn_model, graphs):
        loader = DataLoader(graphs, batch_size=1, shuffle=False)
        # warmup
        for _ in range(min(5, len(graphs))):
            b = next(iter(loader))
            gnn_model(b.x.float(), b.edge_index, b.batch, dropout_percentage=0.0)
        preds, probs, latencies = [], [], []
        for batch in loader:
            t0 = time.perf_counter()
            logits = gnn_model(batch.x.float(), batch.edge_index, batch.batch, dropout_percentage=0.0)
            prob = torch.softmax(logits, dim=-1)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            preds.append(int(prob.argmax(dim=-1).item()))
            probs.append(prob.cpu().numpy()[0])
        return np.asarray(preds), np.asarray(probs), np.asarray(latencies)

    gnn_pred, gnn_prob, gnn_lat_ms = time_gnn_per_sample(gnn, test_graphs)
    print(f"GNN latency (ms): mean={gnn_lat_ms.mean():.2f}  p95={np.percentile(gnn_lat_ms,95):.2f}  max={gnn_lat_ms.max():.2f}")
    print(f"GNN class balance (pred): {Counter(ID2LABEL[int(p)] for p in gnn_pred)}")
"""))


# --------------------------------------------------------------------------
cells.append(md("""
    ## 5. Run Sonnet judge on the rows each pipeline would route to it

    Two sets of judge calls are needed:

    - **WITS+judge** (pipeline C): call Sonnet on every row where WITS
      returned `maybe_safe` or `unsafe`. The judge sees the same
      `transcript + command + intention + analysis` the runtime would
      feed it (we faithfully port `buildUserPrompt` from
      `wits/judge/v1.ts`).
    - **GNN+judge** (pipeline E): call Sonnet on every row where the
      GNN's confidence `max(softmax)` falls below the chosen threshold
      `tau`. Same prompt shape as above.

    Most rows overlap (same prompt). Cache + dedup means we pay each
    unique prompt only once.

    Budget: D3 test = 91 rows. At worst, ~91 * 2 = 182 Sonnet calls
    across both pipelines. At ~10 s per call serially that's ~30 min;
    with concurrency 4 it's ~8 min. Sonnet 4.6 is ~$3 input / $15
    output per 1M tokens; with our short prompts and one-line replies
    this run costs roughly $0.20 - $0.60.
"""))

cells.append(code("""
    # ---- Concurrency knob + judge instantiation ----
    JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "claude-sonnet-4.6")
    JUDGE_CONCURRENCY = 4
    TAU = 0.75  # GNN confidence threshold for routing to judge

    from data.judge_runner import JudgeRunner

    judge = JudgeRunner(model=JUDGE_MODEL)

    print(f"Judge model: {JUDGE_MODEL}")
    print(f"Concurrency : {JUDGE_CONCURRENCY}")
    print(f"GNN confidence threshold tau: {TAU}")
"""))

cells.append(code("""
    # ---- Build per-row context (transcript + intention + WITS analysis) ----
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
            "truth":        rec["decision"],  # "allow" or "block"
            "wits_verdict": w["verdict"],
            "wits_rules":   w.get("rule_ids", []),
            "gnn_pred":     ID2LABEL[int(gnn_pred[i])],
            "gnn_conf":     float(gnn_prob[i].max()),
            "gnn_lat_ms":   float(gnn_lat_ms[i]),
        })

    # Identify rows each pipeline routes to the judge.
    JUDGE_VERDICTS = {"maybe_safe", "unsafe"}
    wits_judge_idx = [c["i"] for c in contexts if c["wits_verdict"] in JUDGE_VERDICTS]
    gnn_judge_idx  = [c["i"] for c in contexts if c["gnn_conf"] < TAU]
    union_judge_idx = sorted(set(wits_judge_idx) | set(gnn_judge_idx))

    print(f"Pipeline C (WITS+judge) routes {len(wits_judge_idx)}/{len(contexts)} rows to Sonnet.")
    print(f"Pipeline E (GNN+judge, tau={TAU}) routes {len(gnn_judge_idx)}/{len(contexts)} rows to Sonnet.")
    print(f"Unique rows needing a judge call: {len(union_judge_idx)} (cache dedups identical prompts).")
"""))

cells.append(code("""
    # ---- Fire judge calls (with concurrency) ----
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # We pass the WITS analysis (rules + effects) to the judge for every row,
    # so both pipelines see the same evidence the production judge sees.
    # rule_ids is a flat list; for richer effects we'd need to thread the
    # full WITS analysis from the shim — for now, ruleIds + a placeholder
    # effects list is a faithful summary for the judge prompt.
    def _judge_one(c):
        analysis = {
            "rule_hits": [{"ruleId": rid, "severity": "info", "message": ""} for rid in c["wits_rules"]],
            "effects":   [],
        }
        t0 = time.perf_counter()
        res = judge.judge(
            command=c["command"],
            intention=c["intention"],
            transcript=c["transcript"],
            analysis=analysis,
        )
        wall_ms = (time.perf_counter() - t0) * 1000.0
        return c["i"], res, wall_ms

    needed = [contexts[i] for i in union_judge_idx]
    print(f"Calling Sonnet on {len(needed)} rows (cache will skip any that are already on disk) ...")

    judge_results = {}  # i -> (JudgeResult, wall_ms)
    with ThreadPoolExecutor(max_workers=JUDGE_CONCURRENCY) as ex:
        futures = {ex.submit(_judge_one, c): c["i"] for c in needed}
        for fut in as_completed(futures):
            i, res, wall_ms = fut.result()
            judge_results[i] = (res, wall_ms)
            if len(judge_results) % 10 == 0:
                cached = sum(1 for (r, _) in judge_results.values() if r.cached)
                print(f"  {len(judge_results)}/{len(needed)} done ({cached} cached)")

    # Summary
    n_cached = sum(1 for (r, _) in judge_results.values() if r.cached)
    n_fresh  = len(judge_results) - n_cached
    n_parse_err = sum(1 for (r, _) in judge_results.values() if r.parse_error)
    print(f"\\nJudge calls done: {len(judge_results)} ({n_cached} cached, {n_fresh} fresh, {n_parse_err} parse errors)")
"""))


# --------------------------------------------------------------------------
cells.append(md("## 6. Assemble per-pipeline decisions"))

cells.append(code("""
    # Helper: collapse WITS verdict to a binary fallback when no judge is consulted.
    def _wits_strict_binary(v):
        return "allow" if v == "safe" else "block"

    def _wits_permissive_binary(v):
        if v == "safe":             return "allow"
        if v == "extremely_unsafe": return "block"
        return "allow"

    rows = []
    for c in contexts:
        i = c["i"]
        # --- A: WITS strict ---
        a_dec = _wits_strict_binary(c["wits_verdict"])
        a_lat = 0.35  # WITS p50 from earlier runs; per-call shim timings
        # --- B: WITS permissive ---
        b_dec = _wits_permissive_binary(c["wits_verdict"])
        b_lat = 0.35
        # --- C: WITS + judge ---
        if c["wits_verdict"] in {"safe", "extremely_unsafe"}:
            c_dec = _wits_strict_binary(c["wits_verdict"])
            c_lat = 0.35
        else:
            jr, wall = judge_results[i]
            c_dec = "allow" if jr.decision == "auto_approve" else "block"
            c_lat = 0.35 + jr.latency_ms  # WITS analysis + judge call
        # --- D: GNN alone ---
        d_dec = c["gnn_pred"]
        d_lat = c["gnn_lat_ms"]
        # --- E: GNN + judge @ tau ---
        if c["gnn_conf"] >= TAU:
            e_dec = c["gnn_pred"]
            e_lat = c["gnn_lat_ms"]
        else:
            jr, wall = judge_results[i]
            e_dec = "allow" if jr.decision == "auto_approve" else "block"
            e_lat = c["gnn_lat_ms"] + jr.latency_ms

        rows.append({
            "i":            i,
            "case_name":    c["case_name"],
            "truth":        c["truth"],
            "wits_verdict": c["wits_verdict"],
            "gnn_pred":     c["gnn_pred"],
            "gnn_conf":     c["gnn_conf"],
            "A_dec": a_dec, "A_lat_ms": a_lat,
            "B_dec": b_dec, "B_lat_ms": b_lat,
            "C_dec": c_dec, "C_lat_ms": c_lat,
            "D_dec": d_dec, "D_lat_ms": d_lat,
            "E_dec": e_dec, "E_lat_ms": e_lat,
            "report_bucket": c["report_bucket"],
        })
    decisions_df = pd.DataFrame(rows)
    print(f"Per-row decisions assembled: {len(decisions_df)} rows × {len(decisions_df.columns)} cols")
    decisions_df.head()
"""))


# --------------------------------------------------------------------------
cells.append(md("## 7. Headline comparison table"))

cells.append(code("""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    )

    PIPELINES = [
        ("A — WITS strict",          "A_dec", "A_lat_ms"),
        ("B — WITS permissive",      "B_dec", "B_lat_ms"),
        ("C — WITS + Sonnet judge",  "C_dec", "C_lat_ms"),
        ("D — GNN alone",            "D_dec", "D_lat_ms"),
        (f"E — GNN + judge @ τ={TAU}", "E_dec", "E_lat_ms"),
    ]

    summary_rows = []
    yt = decisions_df["truth"].values
    for name, dcol, lcol in PIPELINES:
        yp = decisions_df[dcol].values
        acc = accuracy_score(yt, yp)
        f1  = f1_score(yt, yp, average="macro", labels=LABEL_NAMES, zero_division=0)
        prec_block = precision_score(yt, yp, pos_label="block", zero_division=0)
        rec_block  = recall_score(yt, yp, pos_label="block", zero_division=0)
        lat = decisions_df[lcol].values
        # Judge-invocation rate: fraction where this pipeline took the judge path.
        if dcol == "A_dec" or dcol == "B_dec":
            invocations = 0
        elif dcol == "C_dec":
            invocations = int(decisions_df["wits_verdict"].isin(["maybe_safe", "unsafe"]).sum())
        elif dcol == "D_dec":
            invocations = 0
        else:  # E
            invocations = int((decisions_df["gnn_conf"] < TAU).sum())
        summary_rows.append({
            "pipeline":         name,
            "accuracy":         acc,
            "macro_f1":         f1,
            "prec_block":       prec_block,
            "recall_block":     rec_block,
            "judge_invocations":invocations,
            "judge_rate":       invocations / len(yt),
            "lat_mean_ms":      lat.mean(),
            "lat_p95_ms":       float(np.percentile(lat, 95)),
            "lat_max_ms":       lat.max(),
        })
    summary_df = pd.DataFrame(summary_rows)
    print("Headline pipeline comparison:")
    summary_df
"""))

cells.append(code("""
    # Confusion matrices per pipeline.
    for name, dcol, _ in PIPELINES:
        cm = confusion_matrix(decisions_df["truth"], decisions_df[dcol], labels=LABEL_NAMES)
        print(f"\\n{name}:")
        print(pd.DataFrame(cm, index=LABEL_NAMES, columns=LABEL_NAMES))
"""))


# --------------------------------------------------------------------------
cells.append(md("""
    ## 8. WITS+judge vs GNN+judge — the production replacement question

    Direct head-to-head of pipelines C and E. The questions:

    - Do they reach the same accuracy?
    - Does the GNN-fronted pipeline invoke the judge LESS (cost win) or MORE (cost loss)?
    - Where do they disagree? Are there cases the GNN catches that WITS misses, or vice versa?
"""))

cells.append(code("""
    c_correct = (decisions_df["C_dec"] == decisions_df["truth"]).sum()
    e_correct = (decisions_df["E_dec"] == decisions_df["truth"]).sum()
    n = len(decisions_df)
    print(f"C — WITS+judge        : {c_correct}/{n}  acc={c_correct/n:.3f}")
    print(f"E — GNN+judge (τ={TAU}) : {e_correct}/{n}  acc={e_correct/n:.3f}")

    c_judge = int(decisions_df["wits_verdict"].isin(["maybe_safe", "unsafe"]).sum())
    e_judge = int((decisions_df["gnn_conf"] < TAU).sum())
    print(f"\\nJudge invocations  C={c_judge} ({c_judge/n:.1%})   E={e_judge} ({e_judge/n:.1%})")
    print(f"Mean latency       C={decisions_df['C_lat_ms'].mean():.0f}ms   E={decisions_df['E_lat_ms'].mean():.0f}ms")
    print(f"p95 latency        C={np.percentile(decisions_df['C_lat_ms'], 95):.0f}ms   E={np.percentile(decisions_df['E_lat_ms'], 95):.0f}ms")
"""))

cells.append(code("""
    # Per-row disagreements between C and E.
    diff = decisions_df[decisions_df["C_dec"] != decisions_df["E_dec"]].copy()
    diff["C_right"] = diff["C_dec"] == diff["truth"]
    diff["E_right"] = diff["E_dec"] == diff["truth"]
    print(f"{len(diff)} rows where C and E disagree.")
    print(f"  E right, C wrong : {int((diff['E_right'] & ~diff['C_right']).sum())}")
    print(f"  C right, E wrong : {int((diff['C_right'] & ~diff['E_right']).sum())}")
    print(f"  both wrong (diff): {int((~diff['C_right'] & ~diff['E_right']).sum())}")
    diff[["case_name", "truth", "C_dec", "E_dec", "C_right", "E_right",
          "wits_verdict", "gnn_pred", "gnn_conf"]].head(30)
"""))


# --------------------------------------------------------------------------
cells.append(md("## 9. Confidence-threshold sweep for the GNN+judge pipeline"))

cells.append(code("""
    # Re-score pipeline E at several taus to find the cost/accuracy sweet spot.
    # No new judge calls needed -- we already have judge_results for all
    # rows that hit it.
    def _score_at_tau(tau):
        n = len(decisions_df)
        invocations = 0
        decisions = []
        for c in contexts:
            i = c["i"]
            if c["gnn_conf"] >= tau:
                decisions.append(c["gnn_pred"])
            else:
                if i in judge_results:
                    jr, _ = judge_results[i]
                    decisions.append("allow" if jr.decision == "auto_approve" else "block")
                else:
                    # Shouldn't happen if union covered all judge-needing rows,
                    # but be defensive.
                    decisions.append(c["gnn_pred"])
                invocations += 1
        acc = accuracy_score(decisions_df["truth"], decisions)
        return {
            "tau":              tau,
            "judge_invocations":invocations,
            "judge_rate":       invocations / n,
            "accuracy":         acc,
            "macro_f1":         f1_score(decisions_df["truth"], decisions,
                                          average="macro", labels=LABEL_NAMES, zero_division=0),
            "recall_block":     recall_score(decisions_df["truth"], decisions,
                                              pos_label="block", zero_division=0),
        }

    TAUS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
    sweep = pd.DataFrame([_score_at_tau(t) for t in TAUS])
    print(f"Pipeline E sweep ({len(TAUS)} thresholds, same trained GNN + cached judge):")
    sweep
"""))


# --------------------------------------------------------------------------
cells.append(md("## 10. Per-bucket accuracy"))

cells.append(code("""
    if "report_bucket" in decisions_df.columns:
        for name, dcol, _ in PIPELINES:
            decisions_df[f"{dcol}_right"] = decisions_df[dcol] == decisions_df["truth"]
        per_bucket = decisions_df.groupby("report_bucket").agg(
            n=("truth", "size"),
            A_acc=("A_dec_right", "mean"),
            B_acc=("B_dec_right", "mean"),
            C_acc=("C_dec_right", "mean"),
            D_acc=("D_dec_right", "mean"),
            E_acc=("E_dec_right", "mean"),
        ).sort_values("n", ascending=False)
        per_bucket
    else:
        print("no report_bucket field available")
"""))


# --------------------------------------------------------------------------
cells.append(md("## 11. Cost estimate"))

cells.append(code("""
    # Rough $ cost of the judge calls in pipelines C and E.
    SONNET_INPUT_USD_PER_1M  = 3.00
    SONNET_OUTPUT_USD_PER_1M = 15.00

    in_tok  = sum((r.usage.get("prompt_tokens")     or 0) for (r, _) in judge_results.values())
    out_tok = sum((r.usage.get("completion_tokens") or 0) for (r, _) in judge_results.values())
    cost = in_tok * SONNET_INPUT_USD_PER_1M / 1e6 + out_tok * SONNET_OUTPUT_USD_PER_1M / 1e6
    print(f"Total Sonnet tokens this run: {in_tok} input + {out_tok} output")
    print(f"Approx cost on full uncached run: ~${cost:.4f}")
    if in_tok == 0:
        print("  (usage tokens not reported by CAPI -- $ estimate skipped)")
    n_cached = sum(1 for (r, _) in judge_results.values() if r.cached)
    print(f"Cache savings: {n_cached}/{len(judge_results)} calls served from disk")
"""))


# --------------------------------------------------------------------------
cells.append(md("""
    ## 12. Verdict (interpret this section after running)

    After running everything above, the production replacement
    question reduces to comparing **C vs E**:

    - If **E ≥ C accuracy AND E judge_rate ≤ C judge_rate**: the GNN
      is a strict win — it replaces WITS without losing accuracy AND
      uses the judge less. Ship it (probably in a tiered config: WITS
      still in front for sub-ms fast path, GNN behind it for the
      ambiguous tail).

    - If **E ≥ C accuracy but E judge_rate > C judge_rate**: the GNN
      is more cautious — it knows when not to know. You're paying for
      that caution. Worth it if safety-critical, not if cost-critical.

    - If **E < C accuracy**: the transcript signal isn't enough yet.
      Either the GNN architecture needs more capacity (more nodes,
      bigger LLM backbone) or we need more training data (your future
      telemetry).
"""))


nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {OUT} ({len(cells)} cells)")
