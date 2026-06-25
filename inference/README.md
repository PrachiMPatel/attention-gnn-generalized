# WITS GNN Inference Package

Self-contained inference package for the **attention-graph GNN** that replaces
the WITS static shell-command classifier. Built for platform-engineer handoff.

The GNN takes a shell command + shell name and emits a 4-class verdict
(`safe` / `maybe_safe` / `unsafe` / `extremely_unsafe`) plus a calibrated
confidence. The intended deployment is as the **static layer** of the
auto-approve pipeline, replacing the existing 15-20k-line WITS rule engine.

See `outputs/wits_gnn_comparison_report.docx` in the parent repo for the
full evaluation. Headline: GNN reaches **0.900 3-class accuracy** vs WITS's
**0.775** on the 311-row test split (same input contract), with **~5x lower
silent auto-approve rate**.

## What's in here

```
inference/
├── README.md                      ← this file
├── requirements.txt               ← pinned versions
├── __init__.py
├── gnn_inference.py               ← WitsGnnClassifier, featurizer, routing helper
├── graph_classifier.py            ← GATv2 model definition (verbatim copy)
├── sample_inference.ipynb         ← worked example for the platform engineer
└── models/
    ├── gnn_wits_v1.pt             ← trained GNN state-dict (~1 MB)
    └── gnn_wits_v1_metadata.json  ← hidden dims, num_classes, class weights
```

The featurizer is the **frozen** `Qwen/Qwen2.5-0.5B-Instruct` (downloaded on
first run from HuggingFace; no checkpoint of ours, never fine-tuned).

## Quick start

```python
from inference import WitsGnnClassifier, route_decision

clf = WitsGnnClassifier()  # loads featurizer + GNN; takes ~5s on first call
result = clf.classify("rm -rf /tmp/build", shell="bash")
# {
#   "verdict":        "maybe_safe",
#   "verdict_3class": "judge",
#   "confidence":     0.74,
#   "probabilities":  {"safe": 0.08, "maybe_safe": 0.74, "unsafe": 0.15, ...},
#   "latency_ms": {"featurize_ms": 950.2, "gnn_ms": 1.4, "total_ms": 951.6},
# }
decision = route_decision(result)
# "fast_pass" | "route_to_judge" | "hard_deny"
```

## Pipeline integration

The 3-class collapse maps 1:1 to the auto-approve pipeline's routing buckets:

| GNN 4-class verdict | 3-class | Pipeline action |
|---|---|---|
| `safe` (confident) | `safe` | **fast-pass** — auto-approve, do not call LLM judge |
| `maybe_safe` | `judge` | route to LLM judge with conversation context |
| `unsafe` | `judge` | route to LLM judge with conversation context |
| `extremely_unsafe` (confident) | `extremely_unsafe` | **hard-deny** — refuse, do not call LLM judge |
| any verdict, low confidence | (any) | route to LLM judge |

`route_decision()` implements this. Default confidence threshold is **0.85**
for both fast-pass and hard-deny; tune against shadow telemetry before
flipping production routing.

## Setup

```bash
pip install -r requirements.txt
```

`Qwen/Qwen2.5-0.5B-Instruct` (~1 GB) is downloaded from HuggingFace on
first instantiation of `WitsGnnClassifier` and cached under
`~/.cache/huggingface/hub/`.

## Performance characteristics

Measured on CPU, no GPU (Intel laptop, single-threaded inference):

| Stage | Mean | p95 |
|---|---|---|
| Featurization (Qwen forward pass) | ~960 ms | ~1170 ms |
| GNN forward pass | ~1 ms | ~2 ms |
| **End-to-end per command** | **~961 ms** | **~1172 ms** |

On GPU the featurizer drops by ~10x. The GNN itself is negligible.

For an inline command-gating slot that needs <50 ms, run the featurizer on
GPU; CPU is fine for batch / shadow / audit use cases. The GNN forward
pass is essentially free either way.

## Recommended deployment plan

1. **Shadow mode (2-4 weeks).** Log GNN verdict + confidence alongside the
   existing WITS verdict on every command. Do NOT change production routing.
2. **Review the disagreement telemetry.** Confirm the test-set numbers
   hold up on live agent traffic.
3. **Flip routing.** GNN verdict drives the static-layer decision; WITS
   stays as a logged shadow signal for one more cycle, then is removed.

## Caveats / non-goals

This is a **command-pattern classifier**. It does NOT address:

- Untrusted-source prompt injection (hostile repo content, MCP outputs,
  skill SKILL.md files, agent-self-authored scripts). Same blind spot as
  WITS; needs separate provenance work.
- Out-of-workspace writes (`~/.ssh`, `~/.aws`, etc.). Needs explicit
  path-scoping in the pipeline, not a smarter classifier.
- Decoded-from-encoded commands (base64 payloads the agent decodes and
  re-runs). Needs a "decoded-by-agent" pipeline marker.

These are pipeline-architecture concerns that the GNN swap is orthogonal
to. See `outputs/wits_gnn_comparison_report.docx` "Scope and non-goals"
for the full discussion.

## Versioning

| File | Version | Trained | Notes |
|---|---|---|---|
| `models/gnn_wits_v1.pt` | v1 | 2026-06-24 | 4-class, class-weighted CE, GATv2 ×2, hidden [896,128,64] |

Retraining replaces `gnn_wits_v1.pt` + `gnn_wits_v1_metadata.json`. The
featurizer (Qwen 2.5 0.5B Instruct) is frozen and pulled from HuggingFace
at runtime — there is no checkpoint of it to track here.

## Contact

For questions on the model or eval methodology: Prachi Patel (this repo).
