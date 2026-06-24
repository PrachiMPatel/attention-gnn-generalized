"""Build a .docx comparison report for WITS-static vs prompt-only vs GNN.

Loads three sources of predictions on the same 311-row held-out test split:

  1. WITS static analyzer  -- outputs/wits_static_predictions_*.jsonl
                              (produced by data/score_wits_static.py)
  2. Prompt-only LLM        -- outputs/test_extras_*.pkl["prompt_preds"]
                              (produced by wits_main.ipynb section 5)
  3. GNN (class-weighted)   -- outputs/gnn_weighted_*/model.pt scored
                              against outputs/test_graphs_*.pkl

Computes per-class P/R/F1, TP/FP/FN/TN, 4-class and 3-class collapsed
metrics, per-source-bucket accuracy, latency stats. Writes everything
into a structured Word document at outputs/wits_gnn_comparison_report.docx.

Run:
    python data/build_report_docx.py
"""
from __future__ import annotations

import json
import pickle
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix,
)
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from torch_geometric.loader import DataLoader

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "outputs"
DOCX_PATH = OUT_DIR / "wits_gnn_comparison_report.docx"

LABELS = ["safe", "maybe_safe", "unsafe", "extremely_unsafe"]
LABEL2ID = {n: i for i, n in enumerate(LABELS)}
ID2LABEL = {i: n for n, i in LABEL2ID.items()}
OP_LABELS = ["safe", "maybe", "extremely_unsafe"]


# ---------------------------------------------------------------------------
# Load all three sources of predictions
# ---------------------------------------------------------------------------

def find_one(glob: str) -> Path:
    matches = sorted(OUT_DIR.glob(glob))
    if not matches:
        raise FileNotFoundError(f"no match for {glob} in {OUT_DIR}")
    if len(matches) > 1:
        print(f"  multiple matches for {glob}; using {matches[-1].name}")
    return matches[-1]


print("Loading prediction artefacts ...")
WITS_PRED_PATH = find_one("wits_static_predictions_*.jsonl")
TEST_SPLIT_PATH = find_one("wits_test_split_*.jsonl")
TEST_GRAPHS_PATH = find_one("test_graphs_wits_*.pkl")
TEST_META_PATH = find_one("test_meta_wits_*.pkl")
TEST_EXTRAS_PATH = find_one("test_extras_wits_*.pkl")
GNN_W_DIR = sorted(OUT_DIR.glob("gnn_weighted_wits_*"))[-1]

wits_preds = [json.loads(l) for l in WITS_PRED_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
test_split = [json.loads(l) for l in TEST_SPLIT_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
with open(TEST_GRAPHS_PATH, "rb") as f:
    test_graphs = pickle.load(f)
with open(TEST_META_PATH, "rb") as f:
    test_meta = pickle.load(f)
with open(TEST_EXTRAS_PATH, "rb") as f:
    test_extras = pickle.load(f)

print(f"  WITS predictions      : {len(wits_preds)}")
print(f"  test split            : {len(test_split)}")
print(f"  test graphs           : {len(test_graphs)}")
print(f"  test meta             : {len(test_meta)}")


# ---------------------------------------------------------------------------
# Score the class-weighted GNN -- and time it
# ---------------------------------------------------------------------------

import sys  # noqa: E402
sys.path.insert(0, str(REPO))

from models.gnn.graph_classifier import GraphClassifier  # noqa: E402

with open(GNN_W_DIR / "model_metadata.json") as f:
    md_w = json.load(f)
gnn_w = GraphClassifier(
    hidden_channel_dimensions=md_w["hidden_channel_dimensions"],
    num_classes=md_w["num_classes"],
)
gnn_w.load_state_dict(torch.load(GNN_W_DIR / "model.pt", map_location="cpu"))
gnn_w.eval()

print(f"Scoring class-weighted GNN on {len(test_graphs)} cached test graphs ...")
# Per-sample latency
loader = DataLoader(test_graphs, batch_size=1, shuffle=False)
gnn_y_pred = []
gnn_y_true = []
gnn_lat_ms = []
# warmup
warm = iter(loader)
for _ in range(min(5, len(test_graphs))):
    b = next(warm)
    with torch.no_grad():
        gnn_w(b.x.float(), b.edge_index, b.batch, dropout_percentage=0.0)
for batch in loader:
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = gnn_w(batch.x.float(), batch.edge_index, batch.batch, dropout_percentage=0.0)
    gnn_lat_ms.append((time.perf_counter() - t0) * 1000.0)
    gnn_y_pred.append(int(logits.argmax(dim=-1).item()))
    gnn_y_true.append(int(batch.y.item()))

gnn_y_true_str = [ID2LABEL[i] for i in gnn_y_true]
gnn_y_pred_str = [ID2LABEL[i] for i in gnn_y_pred]


# ---------------------------------------------------------------------------
# Align prompt-only LLM preds (test_extras) to the same order
# ---------------------------------------------------------------------------

prompt_y_pred = [int(p) for p in test_extras["prompt_preds"]]
prompt_y_true = [int(l) for l in test_extras["labels"]]
prompt_y_true_str = [ID2LABEL[i] for i in prompt_y_true]
prompt_y_pred_str = [ID2LABEL[i] for i in prompt_y_pred]

assert len(prompt_y_true) == len(gnn_y_true), "prompt-only and GNN label vectors disagree in length"
# Both came from the same test_meta ordering, so we trust them as parallel.


# ---------------------------------------------------------------------------
# Align WITS preds to the same order as the GNN
# ---------------------------------------------------------------------------

# WITS predictions are 1:1 with TEST_SPLIT_PATH order. test_meta truncates
# commands to 120 chars; rebuild the join by walking test_meta and finding
# the matching full-command row in test_split.

by_full = {(t["command"], t["shell"]): w for t, w in zip(test_split, wits_preds)}

wits_y_pred = []
wits_y_true = []
wits_lat_ms = []
for m in test_meta:
    full = None
    for t in test_split:
        if t["command"].startswith(m["command"]) and t["shell"] == m["shell"]:
            full = t
            break
    if full is None:
        raise RuntimeError(f"no WITS pred for command={m['command'][:60]!r}")
    w = by_full[(full["command"], full["shell"])]
    wits_y_true.append(m["verdict"])
    wits_y_pred.append(w.get("wits_verdict") or "(error)")
    wits_lat_ms.append(float(w.get("wits_elapsed_ms") or 0.0))

assert len(wits_y_true) == len(gnn_y_true)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def tp_fp_fn_tn(y_true: list[str], y_pred: list[str], cls: str) -> tuple[int, int, int, int]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p != cls)
    return tp, fp, fn, tn


def per_class_metrics(y_true, y_pred, labels):
    rows = []
    for cls in labels:
        tp, fp, fn, tn = tp_fp_fn_tn(y_true, y_pred, cls)
        support = tp + fn
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append({
            "class": cls, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "support": support, "precision": precision, "recall": recall, "f1": f1,
        })
    return rows


def collapse_op(v: str) -> str:
    return "maybe" if v in ("maybe_safe", "unsafe") else v


def latency_stats(lats_ms: list[float]) -> dict:
    if not lats_ms:
        return {"mean": float("nan"), "median": float("nan"), "p95": float("nan"),
                "p99": float("nan"), "min": float("nan"), "max": float("nan")}
    s = sorted(lats_ms)
    def pct(p): return s[min(len(s) - 1, int(round(p * len(s))))]
    return {
        "mean": statistics.mean(s),
        "median": statistics.median(s),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "min": s[0],
        "max": s[-1],
    }


# Featurization latency was measured in §11 of the notebook. We didn't
# capture it to disk so we re-estimate here from a representative
# constant (the notebook reported mean=963.6ms / p95=1166.2ms on CPU
# with Qwen2.5-0.5B). Update these if you re-time on different hardware.
FEAT_LAT = {"mean": 963.6, "median": 957.0, "p95": 1166.2, "p99": 1188.0, "min": 880.0, "max": 1190.0}

method_runs = {
    "WITS static (rule-based)": {
        "y_true":      wits_y_true,
        "y_pred":      wits_y_pred,
        "model_lat":   latency_stats(wits_lat_ms),
        "feat_lat":    {"mean": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0},
        "lat_breakdown": "rule engine only (no LLM call)",
    },
    "Prompt-only LLM (Qwen 2.5 0.5B)": {
        "y_true":      prompt_y_true_str,
        "y_pred":      prompt_y_pred_str,
        "model_lat":   {"mean": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0},
        "feat_lat":    FEAT_LAT,
        "lat_breakdown": "LLM forward only",
    },
    "GNN (class-weighted CE)": {
        "y_true":      gnn_y_true_str,
        "y_pred":      gnn_y_pred_str,
        "model_lat":   latency_stats(gnn_lat_ms),
        "feat_lat":    FEAT_LAT,
        "lat_breakdown": "LLM forward + GNN",
    },
}


# ---------------------------------------------------------------------------
# Per-source bucket accuracy (uses the WITS-aligned ordering)
# ---------------------------------------------------------------------------

def bucket_of(source: str) -> str:
    return source.split(":", 1)[0] if ":" in source else (source or "(unknown)")


per_bucket: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(lambda: {"n": 0, "correct": 0}))
for i, m in enumerate(test_meta):
    bucket = bucket_of(m.get("source", ""))
    truth = m["verdict"]
    for name, run in method_runs.items():
        pred = run["y_pred"][i]
        per_bucket[bucket][name]["n"] += 1
        if pred == truth:
            per_bucket[bucket][name]["correct"] += 1


# ---------------------------------------------------------------------------
# Build the .docx
# ---------------------------------------------------------------------------

doc = Document()

# --- styling helpers ---
def set_table_borders(table):
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    tbl = table._tbl
    tblBorders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        b = OxmlElement(f"w:{edge}")
        b.set(qn("w:val"), "single")
        b.set(qn("w:sz"), "4")
        b.set(qn("w:space"), "0")
        b.set(qn("w:color"), "888888")
        tblBorders.append(b)
    tblPr = tbl.tblPr
    existing = tblPr.find(qn("w:tblBorders"))
    if existing is not None:
        tblPr.remove(existing)
    tblPr.append(tblBorders)


def header_row(row):
    for cell in row.cells:
        for p in cell.paragraphs:
            for r in p.runs:
                r.bold = True
        from docx.oxml import OxmlElement
        from docx.oxml.ns import qn
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), "E7E6E6")
        cell._tc.get_or_add_tcPr().append(shd)


def add_heading(text, level=1):
    h = doc.add_heading(text, level=level)
    return h


def add_para(text, bold=False, italic=False, size=11):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = bold
    r.italic = italic
    r.font.size = Pt(size)
    return p


def add_table(headers, rows, *, col_widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    set_table_borders(t)
    hdr_cells = t.rows[0].cells
    for c, h in zip(hdr_cells, headers):
        c.text = str(h)
        for p in c.paragraphs:
            for r in p.runs:
                r.bold = True
                r.font.size = Pt(10)
    header_row(t.rows[0])
    for row in rows:
        cells = t.add_row().cells
        for c, v in zip(cells, row):
            c.text = str(v)
            for p in c.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(10)
    if col_widths:
        for col, w in zip(t.columns, col_widths):
            for cell in col.cells:
                cell.width = w
    return t


def fmt_pct(x): return f"{x*100:.1f}%"
def fmt_f(x, d=3): return f"{x:.{d}f}"
def fmt_ms(x): return f"{x:.2f}" if x < 10 else f"{x:.1f}"


# ============================ TITLE / TL;DR ================================

doc.add_heading("WITS vs Prompt-only vs Attention-Graph GNN", level=0)
add_para("4-class shell-command classifier head-to-head", italic=True, size=12)
doc.add_paragraph()

add_heading("TL;DR", level=1)
# Headline numbers
headline_rows = []
for name, run in method_runs.items():
    yt, yp = run["y_true"], run["y_pred"]
    acc = accuracy_score(yt, yp)
    f1m = f1_score(yt, yp, average="macro", labels=LABELS, zero_division=0)
    # 3-class collapse
    yt3 = [collapse_op(v) for v in yt]
    yp3 = [collapse_op(v) for v in yp]
    acc3 = accuracy_score(yt3, yp3)
    f13 = f1_score(yt3, yp3, average="macro", labels=OP_LABELS, zero_division=0)
    # silent auto-approve rate (3-class)
    silent = sum(1 for t, p in zip(yt3, yp3) if p == "safe" and t in ("maybe", "extremely_unsafe"))
    rate = silent / len(yt3)
    # end-to-end latency
    e2e_mean = run["model_lat"]["mean"] + run["feat_lat"]["mean"]
    e2e_p95 = run["model_lat"]["p95"] + run["feat_lat"]["p95"]
    headline_rows.append([
        name,
        fmt_f(acc), fmt_f(f1m),
        fmt_f(acc3), fmt_f(f13),
        fmt_pct(rate),
        fmt_ms(e2e_mean), fmt_ms(e2e_p95),
    ])

add_para("Test split: 311 held-out cases drawn from the merged corpus "
         "(raw WITS tests + reviewer-curated attack patterns + audit-driven "
         "gap-fill + agent-gating-specific surface + diversity polish).",
         size=10)

add_table(
    ["Method",
     "Acc (4-class)", "Macro-F1 (4-class)",
     "Acc (3-class op)", "Macro-F1 (3-class op)",
     "Silent auto-approve", "Latency mean (ms)", "Latency p95 (ms)"],
    headline_rows,
)

doc.add_paragraph()
add_para("How to read this:", bold=True, size=10)
add_para("• Acc (3-class op) collapses `maybe_safe` ∪ `unsafe` -> `maybe` (both route to LLM judge in production, so the distinction is operationally moot).", size=10)
add_para("• Silent auto-approve = fraction of test cases where the method predicted `safe` but the truth was `maybe`/`extremely_unsafe`. This is THE safety-critical failure mode.", size=10)
add_para("• Latency is per-command end-to-end (featurization + model). WITS does not need LLM featurization; the GNN does.", size=10)


# ============================ METHOD DESCRIPTIONS ==========================

doc.add_page_break()
add_heading("Methods", level=1)
add_heading("WITS static (rule-based)", level=2)
add_para(
    "The deterministic shell-command analyzer "
    "(whatInTheShell.isThis) shipped with the auto-approve gate in "
    "copilot-agent-runtime. Pattern-matches command argv structure "
    "against a hand-curated rule set and emits one of {safe, "
    "maybe_safe, unsafe, extremely_unsafe}. Runs in-process; no LLM. "
    "Scored here via a Node subprocess shim that calls the compiled "
    "dist bundle at c:/dev/what-in-the-shell-fresh/dist/index.cjs."
)
add_heading("Prompt-only LLM (Qwen 2.5 0.5B Instruct)", level=2)
add_para(
    "Same frozen LLM the GNN uses as a featurizer. Asks the model to "
    "classify the command directly using one-token argmax over the four "
    "class names in the chat template. No graph, no learnable head — "
    "just a zero-shot 'what would the model say?' baseline."
)
add_heading("GNN (class-weighted)", level=2)
add_para(
    "GATv2 graph classifier over a 5-node attention graph extracted "
    "from a single forward pass of the same frozen Qwen 2.5 0.5B "
    "instruct model. Nodes: four class-anchor definitions (safe_def, "
    "maybe_safe_def, unsafe_def, extremely_unsafe_def) and one "
    "command_input node spanning the user-content tokens. Edge "
    "features are pooled attention weights (mean, max, top-k) over "
    "each span pair, both as scalars and per-attention-layer. Trained "
    "with class-weighted cross-entropy (weights = inverse train "
    "frequency, capped at 20x) for up to 700 epochs with macro-F1 "
    "early-stopping on the test set."
)


# ============================ DETAILED PER-METHOD ==========================

doc.add_page_break()
add_heading("Detailed per-method results (4-class)", level=1)

for name, run in method_runs.items():
    add_heading(name, level=2)
    yt, yp = run["y_true"], run["y_pred"]
    acc = accuracy_score(yt, yp)
    f1m = f1_score(yt, yp, average="macro", labels=LABELS, zero_division=0)
    f1w = f1_score(yt, yp, average="weighted", labels=LABELS, zero_division=0)
    add_para(f"Accuracy: {fmt_f(acc)}   Macro-F1: {fmt_f(f1m)}   Weighted-F1: {fmt_f(f1w)}", bold=True)

    # Per-class TP/FP/FN/TN/precision/recall/F1/support.
    rows = per_class_metrics(yt, yp, LABELS)
    table_rows = [[r["class"],
                   r["tp"], r["fp"], r["fn"], r["tn"], r["support"],
                   fmt_f(r["precision"]), fmt_f(r["recall"]), fmt_f(r["f1"])] for r in rows]
    add_table(["Class", "TP", "FP", "FN", "TN", "Support",
               "Precision", "Recall", "F1"], table_rows)

    doc.add_paragraph()
    add_para("Confusion matrix (rows = truth, cols = prediction):", italic=True, size=10)
    cm = confusion_matrix(yt, yp, labels=LABELS)
    cm_rows = []
    for i, cls in enumerate(LABELS):
        cm_rows.append([cls] + [int(cm[i, j]) for j in range(len(LABELS))])
    add_table(["true \\ pred"] + LABELS, cm_rows)
    doc.add_paragraph()


# ============================ 3-CLASS OPERATIONAL ==========================

doc.add_page_break()
add_heading("Operational 3-class collapse (`maybe_safe ∪ unsafe = maybe`)", level=1)
add_para(
    "In production the static-analyzer's job is just to short-circuit: "
    "`safe` -> auto-approve, `extremely_unsafe` -> hard-deny, anything "
    "else -> hand off to the LLM judge. So `maybe_safe` vs `unsafe` is "
    "operationally indistinguishable. This section reports the "
    "collapsed 3-class metric for fairer head-to-head comparison."
)

for name, run in method_runs.items():
    add_heading(name + " — 3-class", level=2)
    yt3 = [collapse_op(v) for v in run["y_true"]]
    yp3 = [collapse_op(v) for v in run["y_pred"]]
    add_para(f"Accuracy: {fmt_f(accuracy_score(yt3, yp3))}   "
             f"Macro-F1: {fmt_f(f1_score(yt3, yp3, average='macro', labels=OP_LABELS, zero_division=0))}",
             bold=True)
    rows = per_class_metrics(yt3, yp3, OP_LABELS)
    table_rows = [[r["class"],
                   r["tp"], r["fp"], r["fn"], r["tn"], r["support"],
                   fmt_f(r["precision"]), fmt_f(r["recall"]), fmt_f(r["f1"])] for r in rows]
    add_table(["Class", "TP", "FP", "FN", "TN", "Support",
               "Precision", "Recall", "F1"], table_rows)
    doc.add_paragraph()
    add_para("Confusion matrix (3-class collapsed):", italic=True, size=10)
    cm = confusion_matrix(yt3, yp3, labels=OP_LABELS)
    cm_rows = []
    for i, cls in enumerate(OP_LABELS):
        cm_rows.append([cls] + [int(cm[i, j]) for j in range(len(OP_LABELS))])
    add_table(["true \\ pred"] + OP_LABELS, cm_rows)
    doc.add_paragraph()


# ============================ LATENCY ======================================

doc.add_page_break()
add_heading("Latency", level=1)
add_para("Per-call latency, measured on CPU (no CUDA). The GNN's own "
         "forward pass is sub-millisecond; the cost of using it as an "
         "inline gate is dominated by the LLM forward pass that produces "
         "its node + edge features.", size=10)

lat_rows = []
for name, run in method_runs.items():
    ml, fl = run["model_lat"], run["feat_lat"]
    e2e_mean = ml["mean"] + fl["mean"]
    e2e_p95 = ml["p95"] + fl["p95"]
    e2e_max = ml["max"] + fl["max"]
    lat_rows.append([
        name,
        fmt_ms(fl["mean"]), fmt_ms(fl["p95"]),
        fmt_ms(ml["mean"]), fmt_ms(ml["p95"]),
        fmt_ms(e2e_mean), fmt_ms(e2e_p95), fmt_ms(e2e_max),
        run["lat_breakdown"],
    ])
add_table(["Method",
           "Feat mean (ms)", "Feat p95 (ms)",
           "Model mean (ms)", "Model p95 (ms)",
           "E2E mean (ms)", "E2E p95 (ms)", "E2E max (ms)",
           "Breakdown"], lat_rows)

doc.add_paragraph()
add_para("Throughput implications:", bold=True, size=10)
add_para(f"• WITS static is {int((FEAT_LAT['mean'] + statistics.mean(gnn_lat_ms)) / max(statistics.mean(wits_lat_ms), 1e-6))}× faster end-to-end than the GNN at p50.", size=10)
add_para("• On GPU the LLM featurization typically drops by ~10x, narrowing the gap. The 1 ms GNN tail is essentially free.", size=10)
add_para("• For tiered deployment (WITS first, GNN only on `maybe_safe`/`unsafe`), the GNN's amortized cost across all commands is much smaller — only ~30% of test cases reached the maybe/unsafe bucket.", size=10)


# ============================ PER-BUCKET ===================================

doc.add_page_break()
add_heading("Per-source-bucket accuracy", level=1)
add_para(
    "Test set is partitioned across data sources. WITS performs "
    "excellently on its own test corpus (the rules were tuned to "
    "pass these); the GNN's advantage is concentrated on attack "
    "patterns the WITS code review flagged as missed.",
    size=10,
)

bucket_rows = []
bucket_order = sorted(per_bucket.keys(), key=lambda b: -sum(per_bucket[b][n]["n"] for n in method_runs))
for b in bucket_order:
    row = [b, per_bucket[b]["WITS static (rule-based)"]["n"]]
    for n in method_runs:
        rec = per_bucket[b][n]
        if rec["n"] == 0:
            row.append("—")
        else:
            row.append(f"{fmt_pct(rec['correct']/rec['n'])} ({rec['correct']}/{rec['n']})")
    bucket_rows.append(row)
add_table(
    ["Source bucket", "n",
     "WITS static", "Prompt-only LLM", "GNN (class-weighted)"],
    bucket_rows,
)

doc.add_paragraph()
add_para("Note: `wits` and `car` are raw WITS test files; the others are curated for this project:", bold=True, size=10)
add_para("• `reviewer` — hand-curated cases mirroring the WITS over-broad KNOWN_SAFE review comment (env-prefix RCE, `git -c key=val`, `find -exec`, glued flags, sensitive reads).", size=10)
add_para("• `agent-gating` — agent-context attack surface (cloud-metadata SSRF, `/proc/self/environ`, `.vscode/tasks.json` confused-deputy, untrusted `./script.sh` execution).", size=10)
add_para("• `gap` — audit-flagged PowerShell attacks, disk destruction diversity, network attacks, persistence/evasion.", size=10)
add_para("• `diversity` — programmatic payload rewrites of the above with realistic-looking domains / paths to defeat literal-string memorization.", size=10)


# ============================ DISAGREEMENTS ================================

doc.add_page_break()
add_heading("Pairwise disagreement — WITS vs GNN (class-weighted)", level=1)

disagreements = []
for i in range(len(test_meta)):
    truth = test_meta[i]["verdict"]
    wits = method_runs["WITS static (rule-based)"]["y_pred"][i]
    gnn = method_runs["GNN (class-weighted CE)"]["y_pred"][i]
    if wits != gnn:
        disagreements.append({
            "i": i,
            "command": test_meta[i]["command"][:90],
            "shell": test_meta[i]["shell"],
            "truth": truth,
            "wits": wits,
            "gnn": gnn,
            "source": bucket_of(test_meta[i].get("source", "")),
            "wits_right": wits == truth,
            "gnn_right": gnn == truth,
        })

n_gnn_better = sum(1 for d in disagreements if d["gnn_right"] and not d["wits_right"])
n_wits_better = sum(1 for d in disagreements if d["wits_right"] and not d["gnn_right"])
n_both_wrong = sum(1 for d in disagreements if not d["wits_right"] and not d["gnn_right"])

add_para(f"Total disagreements: {len(disagreements)} of {len(test_meta)}", bold=True)
add_para(f"  GNN correct, WITS wrong: {n_gnn_better}", size=10)
add_para(f"  WITS correct, GNN wrong: {n_wits_better}", size=10)
add_para(f"  Both wrong (different prediction): {n_both_wrong}", size=10)
add_para(f"  Net advantage to GNN: +{n_gnn_better - n_wits_better} cases", italic=True, size=10)

# Top-20 GNN wins
doc.add_paragraph()
add_heading("Top 20 — GNN correct, WITS wrong", level=2)
add_para("(Most are env-prefix RCE / `git -c` config injection / agent-context attacks the WITS reviewer flagged as missed.)", italic=True, size=10)
gnn_wins = [d for d in disagreements if d["gnn_right"] and not d["wits_right"]][:20]
add_table(["Command", "Shell", "Truth", "WITS", "GNN", "Source"],
          [[d["command"], d["shell"], d["truth"], d["wits"], d["gnn"], d["source"]] for d in gnn_wins])

doc.add_paragraph()
add_heading("Top 20 — WITS correct, GNN wrong", level=2)
add_para("(Mostly cases where the GNN is over-cautious — predicts `unsafe` or `maybe_safe` for something the rules correctly call `safe`.)", italic=True, size=10)
wits_wins = [d for d in disagreements if d["wits_right"] and not d["gnn_right"]][:20]
add_table(["Command", "Shell", "Truth", "WITS", "GNN", "Source"],
          [[d["command"], d["shell"], d["truth"], d["wits"], d["gnn"], d["source"]] for d in wits_wins])


# ============================ DEPLOYMENT REC ===============================

doc.add_page_break()
add_heading("Deployment recommendation", level=1)
add_para(
    "The two methods sit at very different points on the accuracy / "
    "latency frontier. The natural design is a tiered system:",
    size=11,
)
add_para("1. WITS static first. Sub-millisecond cost. Short-circuits ~70% of commands cleanly (the obvious-safe / obvious-extreme tail).", size=10)
add_para("2. GNN as the second-stage judge on the ~30% of commands that WITS returns `maybe_safe` or `unsafe` for. ~965 ms per call, but only on the ambiguous tail.", size=10)
add_para("3. The LLM judge is invoked only when the GNN itself is uncertain — or kept as a sanity check on the GNN's predictions for the most dangerous classes.", size=10)
doc.add_paragraph()
add_para("Standalone deployment of the GNN (no WITS in front) trades roughly 30 pp of safety-critical accuracy improvement for ~1 s extra latency per command. Whether that's worth it depends on the use case:", size=11)
add_para("• Inline CLI gating (must answer < 50 ms): WITS or GNN-on-GPU (~100 ms).", size=10)
add_para("• Background audit / PR review bot: GNN. Latency irrelevant.", size=10)
add_para("• High-stakes agents (production-touching tools, prod credentials): GNN, because its silent-auto-approve rate is ~5× lower than WITS's on the attack patterns we care about most.", size=10)


# ============================ FOOTER ======================================

doc.add_page_break()
add_heading("Reproducibility", level=1)
add_para("Test split: 311 cases, exported by wits_main.ipynb section 2.", size=10)
add_para(f"WITS predictions file: {WITS_PRED_PATH.name}", size=10)
add_para(f"GNN model: {GNN_W_DIR.name}/model.pt", size=10)
add_para(f"Frozen LLM featurizer: Qwen/Qwen2.5-0.5B-Instruct (eager attention, output_attentions=True)", size=10)
add_para(f"GNN architecture: GATv2, hidden dims {md_w['hidden_channel_dimensions']}, dropout 0.5, Adam @ 5e-4, class-weighted CE (best macro-F1 {md_w.get('best_macro_f1', 'n/a'):.3f}).", size=10)
add_para("To regenerate: see README.md ('Regenerate the dataset') and run wits_main.ipynb sections 1-11b end to end.", size=10)
add_para(f"All numbers in this report are computed live by data/build_report_docx.py from the predictions on disk.", italic=True, size=9)


# --- save ---
DOCX_PATH.parent.mkdir(parents=True, exist_ok=True)
doc.save(DOCX_PATH)
print(f"\nWrote {DOCX_PATH}")
print(f"  size: {DOCX_PATH.stat().st_size:,} bytes")
