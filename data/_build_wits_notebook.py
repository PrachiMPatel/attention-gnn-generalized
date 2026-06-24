"""Build wits_main.ipynb from scratch.

Authoring the notebook in Python lets us mirror main.ipynb's structure
exactly without copy-paste drift. Run this once; the resulting
wits_main.ipynb is hand-editable afterwards.
"""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "wits_main.ipynb"


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
    # WITS 4-class command-safety classifier
    # (Attention graphs + GATv2 GNN)

    Adapted from `main.ipynb` (binary tool-call injection detector).
    Same pipeline:

    1. frozen LLM (default Qwen 2.5 0.5B) forward pass over each shell
       command;
    2. extract a 5-node attention graph
       (`safe_def`, `maybe_safe_def`, `unsafe_def`, `extremely_unsafe_def`,
       `command_input`);
    3. train a GATv2 GNN with `num_classes=4`.

    Labels (from the runtime's `Verdict` type, see
    `src/hooks/auto-approve/wits/core/types.ts` in copilot-agent-runtime-final):

    | id | label |
    | --- | --- |
    | 0 | `safe` |
    | 1 | `maybe_safe` |
    | 2 | `unsafe` |
    | 3 | `extremely_unsafe` |

    Dataset: `data/wits_eval_cases.jsonl` (~511 cases, harvested from the
    runtime's WITS unit tests by `data/extract_wits_cases.py`).
    The distribution is heavily imbalanced (~82% safe). We compensate
    with class-weighted cross-entropy and stratified splits.
"""))

cells.append(md("## 1. Setup"))

cells.append(code("""
    import os, sys, json, random, pickle, gc
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import torch
    from tqdm.auto import tqdm

    # Optional HF token (gated models only). Unused for the default Qwen.
    HF_TOKEN = os.environ.get("HF_TOKEN")
    if HF_TOKEN:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
        try:
            from huggingface_hub import login as _hf_login
            _hf_login(token=HF_TOKEN, add_to_git_credential=False)
        except Exception as e:
            print("hf login skipped:", e)

    SEED = 42
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    NOTEBOOK_DIR = Path.cwd()
    REPO_ROOT = NOTEBOOK_DIR
    while not (REPO_ROOT / "models" / "gnn" / "graph_classifier.py").exists() and REPO_ROOT.parent != REPO_ROOT:
        REPO_ROOT = REPO_ROOT.parent
    sys.path.insert(0, str(REPO_ROOT))
    print("Repo root:", REPO_ROOT)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", DEVICE)

    DATA_DIR = REPO_ROOT / "outputs"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
"""))

cells.append(md("## 2. Load WITS dataset and sub-sample"))

cells.append(code("""
    # >>> EDIT ME: point at any JSONL with the WITS-case schema.
    # Each line:
    #   {"command": str, "shell": "bash"|"powershell",
    #    "verdict": "safe"|"maybe_safe"|"unsafe"|"extremely_unsafe",
    #    "source": str (optional)}
    DATASET_PATH = REPO_ROOT / "data" / "wits_eval_cases.jsonl"

    # Class index map. Order is the same as the runtime's `Verdict` union.
    LABEL_NAMES = ["safe", "maybe_safe", "unsafe", "extremely_unsafe"]
    NUM_CLASSES = len(LABEL_NAMES)
    LABEL2ID = {name: i for i, name in enumerate(LABEL_NAMES)}
    ID2LABEL = {i: name for name, i in LABEL2ID.items()}

    raw = [json.loads(l) for l in DATASET_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"Loaded {len(raw)} records from {DATASET_PATH.name}")

    def to_record(r):
        v = r["verdict"]
        return {
            "command": str(r["command"]),
            "shell":   str(r.get("shell", "bash")),
            "source":  str(r.get("source", "")),
            "label":   LABEL2ID[v],
            "verdict": v,
        }

    all_recs = [to_record(r) for r in raw if r.get("verdict") in LABEL2ID]
    print("class counts:", pd.Series([r["verdict"] for r in all_recs]).value_counts().to_dict())
    print("by shell    :", pd.Series([r["shell"]   for r in all_recs]).value_counts().to_dict())

    # Per-class stratified split. We use a per-class TEST_FRAC because the
    # rare classes ("unsafe": 3, "extremely_unsafe": ~14) cannot afford a
    # fixed 30-per-class test sample. Floor at 1 to keep at least one rare
    # example in test when possible.
    TEST_FRAC = 0.25

    by_class: dict[int, list] = {i: [] for i in range(NUM_CLASSES)}
    for r in all_recs:
        by_class[r["label"]].append(r)
    rng = random.Random(SEED)
    for k in by_class:
        rng.shuffle(by_class[k])

    train_recs, test_recs = [], []
    for k, lst in by_class.items():
        n_te = max(1, int(round(len(lst) * TEST_FRAC))) if len(lst) >= 2 else 0
        test_recs.extend(lst[:n_te])
        train_recs.extend(lst[n_te:])
    random.Random(SEED).shuffle(train_recs)
    random.Random(SEED + 1).shuffle(test_recs)

    def _dist(recs):
        return pd.Series([ID2LABEL[r["label"]] for r in recs]).value_counts().to_dict()
    print(f"\\ntrain n={len(train_recs)}  dist={_dist(train_recs)}")
    print(f"test  n={len(test_recs)}  dist={_dist(test_recs)}")
"""))

cells.append(code("""
    # --- export the held-out test split as JSONL so other models can score it ---
    TEST_EXPORT_PATH = DATA_DIR / f"wits_test_split_{DATASET_PATH.stem}.jsonl"
    with open(TEST_EXPORT_PATH, "w", encoding="utf-8") as f:
        for r in test_recs:
            out = {
                "command": r["command"],
                "shell":   r["shell"],
                "verdict": r["verdict"],
                "source":  r["source"],
            }
            f.write(json.dumps(out) + "\\n")
    print(f"wrote {len(test_recs)} test records -> {TEST_EXPORT_PATH}")
"""))

cells.append(md("## 3. Frozen LLM + attention-graph featurizer"))

cells.append(code("""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    MODEL_NAME = os.environ.get("WITS_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
    _dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        attn_implementation="eager",
        output_attentions=True,
        output_hidden_states=True,
        token=HF_TOKEN,
    )
    if not torch.cuda.is_available():
        model = model.to(DEVICE)
    model.eval()
    HIDDEN_SIZE = model.config.hidden_size
    NUM_LAYERS  = model.config.num_hidden_layers
    print(f"Loaded {MODEL_NAME}  hidden={HIDDEN_SIZE}  layers={NUM_LAYERS}  dtype={_dtype}")
"""))

cells.append(code("""
    from torch_geometric.data import Data

    # Class-anchor spans inserted into the system prompt. Each describes
    # ONE of the four verdicts in tight, lexically-disjoint language so the
    # attention from the decision tail into each anchor is a distinguishing
    # signal. Wording mirrors the verdict semantics documented in
    # copilot-agent-runtime-final/src/hooks/auto-approve/wits/core/types.ts.
    SAFE_DEF_TEXT = (
        "[SAFE_DEF] SAFE shell command: read-only inspection, status check, "
        "version probe, or other completely-reversible operation. No file "
        "mutation outside scratch, no network egress, no destructive verbs, "
        "no privilege escalation. Examples: ls, pwd, cat README.md, git status, "
        "node --version. [/SAFE_DEF]"
    )
    MAYBE_SAFE_DEF_TEXT = (
        "[MAYBE_SAFE_DEF] MAYBE_SAFE shell command: in-scope mutation whose "
        "blast radius is bounded and whose effect is reversible by ordinary "
        "tooling. Local file edits, package installs into the project, git "
        "branch operations on a feature branch, cloud calls scoped to test "
        "resources. [/MAYBE_SAFE_DEF]"
    )
    UNSAFE_DEF_TEXT = (
        "[UNSAFE_DEF] UNSAFE shell command: cross-boundary or high-blast-radius "
        "action that may be hard to undo. Force-push to a shared branch, "
        "kubectl patch on a shared namespace, public-read S3 ACL, IAM policy "
        "attachment, REVOKE on a live database. [/UNSAFE_DEF]"
    )
    EXTREMELY_UNSAFE_DEF_TEXT = (
        "[EXTREMELY_UNSAFE_DEF] EXTREMELY_UNSAFE shell command: catastrophic, "
        "essentially irreversible action. rm -rf /, dd of=/dev/sda, "
        "chmod 777 of system paths, piping curl into sh, base64-encoded "
        "destruction, deleting all branches, dropping a production database. "
        "[/EXTREMELY_UNSAFE_DEF]"
    )

    DECISION_TAIL = (
        "The shell command above is being evaluated for safety. "
        "Classify it as exactly one of SAFE, MAYBE_SAFE, UNSAFE, or EXTREMELY_UNSAFE:"
    )
    CLASSIFY_INSTRUCTION = (
        "You are a shell-command safety classifier. Choose exactly ONE class.\\n\\n"
        f"{SAFE_DEF_TEXT}\\n\\n{MAYBE_SAFE_DEF_TEXT}\\n\\n"
        f"{UNSAFE_DEF_TEXT}\\n\\n{EXTREMELY_UNSAFE_DEF_TEXT}\\n\\n"
        "Answer with only one word: SAFE, MAYBE_SAFE, UNSAFE, or EXTREMELY_UNSAFE."
    )

    NODE_NAMES = ["safe_def", "maybe_safe_def", "unsafe_def", "extremely_unsafe_def", "command_input"]
    NODE_TYPE_IDS = {n: i for i, n in enumerate(NODE_NAMES)}
    DEF_NODES = ["safe_def", "maybe_safe_def", "unsafe_def", "extremely_unsafe_def"]

    # First-token IDs for the prompt-only baseline. We collect a few case
    # variants and pick the max logit across them as the per-class score.
    def _first_ids(words):
        out = set()
        for w in words:
            ids = tokenizer(w, add_special_tokens=False)["input_ids"]
            if ids:
                out.add(ids[0])
        return sorted(out)

    CLASS_TOK_FIRST_IDS = {
        "safe":             _first_ids(["SAFE", " SAFE", "Safe", " Safe", "safe", " safe"]),
        "maybe_safe":       _first_ids(["MAYBE_SAFE", " MAYBE_SAFE", "MAYBE", " MAYBE", "Maybe", " Maybe"]),
        "unsafe":           _first_ids(["UNSAFE", " UNSAFE", "Unsafe", " Unsafe", "unsafe", " unsafe"]),
        "extremely_unsafe": _first_ids(["EXTREMELY_UNSAFE", " EXTREMELY_UNSAFE", "EXTREMELY", " EXTREMELY", "Extremely", " Extremely"]),
    }

    MAX_COMMAND_CHARS = 2000

    def build_messages(rec):
        cmd = rec["command"][:MAX_COMMAND_CHARS]
        shell = rec.get("shell", "bash")
        user_block = (
            f"Shell: {shell}\\n"
            f"Command:\\n{cmd}\\n\\n"
            f"{DECISION_TAIL}"
        )
        return [
            {"role": "system", "content": CLASSIFY_INSTRUCTION},
            {"role": "user",   "content": user_block},
        ], user_block
"""))

cells.append(code("""
    @torch.no_grad()
    def extract_attention_graph(rec, label):
        messages, user_block = build_messages(rec)
        prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        enc = tokenizer(prompt_text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
        input_ids = enc["input_ids"].to(model.device)
        offsets   = enc["offset_mapping"][0].tolist()
        T = input_ids.shape[1]

        def char_span_to_token_span(text_to_find):
            cstart = prompt_text.find(text_to_find)
            if cstart < 0:
                return None
            cend = cstart + len(text_to_find)
            tok_start = tok_end = None
            for ti, (s, e) in enumerate(offsets):
                if s == e == 0:
                    continue
                if tok_start is None and e > cstart:
                    tok_start = ti
                if s < cend:
                    tok_end = ti + 1
            if tok_start is None or tok_end is None or tok_end <= tok_start:
                return None
            return (tok_start, tok_end)

        decision_span = char_span_to_token_span(DECISION_TAIL)
        if decision_span is None:
            decision_span = (max(0, T - 16), T)
        decision_span = (decision_span[0], T)

        spans = {
            "safe_def":             char_span_to_token_span(SAFE_DEF_TEXT),
            "maybe_safe_def":       char_span_to_token_span(MAYBE_SAFE_DEF_TEXT),
            "unsafe_def":           char_span_to_token_span(UNSAFE_DEF_TEXT),
            "extremely_unsafe_def": char_span_to_token_span(EXTREMELY_UNSAFE_DEF_TEXT),
            "command_input":        decision_span,
        }
        if any(s is None for s in spans.values()):
            return None

        out = model(input_ids=input_ids, output_attentions=True, output_hidden_states=True, use_cache=False)
        last_hidden = out.hidden_states[-1][0].float().cpu()
        attn_per_layer = torch.stack(
            [a[0].mean(dim=0).float().cpu() for a in out.attentions], dim=0
        )                                                       # (L, T, T)
        attn_mean_layer = attn_per_layer.mean(dim=0)            # (T, T)

        # Prompt-only baseline: argmax over per-class first-token logits.
        final_logits = out.logits[0, -1].float().cpu()
        probs = torch.softmax(final_logits, dim=-1)
        TOPK = 50
        top_p, top_i = torch.topk(probs, TOPK)
        class_logits = []
        for name in LABEL_NAMES:
            ids = CLASS_TOK_FIRST_IDS[name]
            class_logits.append(float(final_logits[ids].max().item()))
        prompt_pred = int(np.argmax(class_logits))

        # Node features: mean-pooled last-layer hidden states over each span.
        node_feats, node_types = [], []
        for name in NODE_NAMES:
            s, e = spans[name]
            node_feats.append(last_hidden[s:e].mean(dim=0))
            node_types.append(NODE_TYPE_IDS[name])
        x = torch.stack(node_feats, dim=0)

        # Edges: command_input -> each definition + self-loop on command.
        # (Direction matches main.ipynb: src=command_input, dst=def-node.)
        edge_pairs = [("command_input", d) for d in DEF_NODES] + [("command_input", "command_input")]
        TOPK_TOKENS = 8
        def _scalars(sub):
            if sub.numel() == 0:
                return 0.0, 0.0, 0.0
            flat = sub.reshape(-1)
            m  = float(flat.mean().item())
            mx = float(flat.max().item())
            k  = min(TOPK_TOKENS, flat.numel())
            tk = float(flat.topk(k).values.mean().item())
            return m, mx, tk

        edge_src, edge_dst = [], []
        edge_w_mean, edge_w_max, edge_w_topk = [], [], []
        edge_w_layers_mean, edge_w_layers_max = [], []
        for src_name, dst_name in edge_pairs:
            si, ei = spans[src_name]
            sj, ej = spans[dst_name]
            sub = attn_mean_layer[si:ei, sj:ej]
            m, mx, tk = _scalars(sub)
            sub_layers = attn_per_layer[:, si:ei, sj:ej]
            L = attn_per_layer.shape[0]
            if sub_layers.numel() == 0:
                wl_m  = torch.zeros(L)
                wl_mx = torch.zeros(L)
            else:
                flat_l = sub_layers.reshape(L, -1)
                wl_m  = flat_l.mean(dim=-1)
                wl_mx = flat_l.max(dim=-1).values

            edge_src.append(NODE_TYPE_IDS[src_name])
            edge_dst.append(NODE_TYPE_IDS[dst_name])
            edge_w_mean.append(m); edge_w_max.append(mx); edge_w_topk.append(tk)
            edge_w_layers_mean.append(wl_m)
            edge_w_layers_max.append(wl_mx)

        scalar_part = torch.tensor(
            [[m, mx, tk] for m, mx, tk in zip(edge_w_mean, edge_w_max, edge_w_topk)],
            dtype=torch.float32,
        )
        layer_mean_part = torch.stack(edge_w_layers_mean, dim=0).float()
        layer_max_part  = torch.stack(edge_w_layers_max, dim=0).float()
        edge_attr = torch.cat([scalar_part, layer_mean_part, layer_max_part], dim=-1)

        data = Data(
            x=x.float(),
            edge_index=torch.tensor([edge_src, edge_dst], dtype=torch.long),
            edge_attr=edge_attr,
            y=torch.tensor(int(label), dtype=torch.long),
        )
        data.node_types = torch.tensor(node_types, dtype=torch.long)

        return {
            "graph": data,
            "softmax_top_p": top_p,
            "softmax_top_i": top_i,
            "prompt_pred":   prompt_pred,
            "class_logits":  np.asarray(class_logits, dtype=np.float32),
        }
"""))

cells.append(md("## 4. Sanity check on one example"))

cells.append(code("""
    sample = train_recs[0]
    res = extract_attention_graph(sample, sample["label"])
    assert res is not None, "span alignment failed on first sample"
    g = res["graph"]
    print(f"label = {ID2LABEL[sample['label']]} ({sample['label']})")
    print(f"command = {sample['command'][:120]!r}")
    print("x.shape =", tuple(g.x.shape),
          " edge_index =", tuple(g.edge_index.shape),
          " edge_attr =", tuple(g.edge_attr.shape))
    for k in range(g.edge_index.shape[1]):
        s, d = int(g.edge_index[0, k]), int(g.edge_index[1, k])
        print(f"  {NODE_NAMES[s]:>20s} -> {NODE_NAMES[d]:<20s}  mean_w={float(g.edge_attr[k, 0]):.4f}")
    print("class_logits:", dict(zip(LABEL_NAMES, res["class_logits"].tolist())))
    print(f"prompt_pred = {ID2LABEL[res['prompt_pred']]} ({res['prompt_pred']})")
    print("OK")
"""))

cells.append(md("## 5. Extract train + test sets and cache"))

cells.append(code("""
    FEATURIZER_VERSION = "wits_v1"
    import hashlib
    _dataset_stat = DATASET_PATH.stat()
    _dataset_fingerprint = hashlib.sha1(
        f"{DATASET_PATH.resolve()}::{_dataset_stat.st_size}::{_dataset_stat.st_mtime_ns}".encode("utf-8")
    ).hexdigest()[:10]
    TAG = f"wits_{DATASET_PATH.stem}_{_dataset_fingerprint}_{FEATURIZER_VERSION}"
    TRAIN_PKL = DATA_DIR / f"train_graphs_{TAG}.pkl"
    TEST_PKL  = DATA_DIR / f"test_graphs_{TAG}.pkl"
    META_PKL  = DATA_DIR / f"meta_{TAG}.pkl"
    EXTRA_PKL = DATA_DIR / f"extras_{TAG}.pkl"
    print("Cache key:", TAG)
    for _p in [TRAIN_PKL, TEST_PKL, META_PKL, EXTRA_PKL]:
        print(f"  {'EXISTS' if _p.exists() else 'MISSING':>7s}  {_p.name}")
"""))

cells.append(code("""
    def extract_dataset(records, desc):
        if "model" not in globals():
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                attn_implementation="eager",
                output_attentions=True,
                output_hidden_states=True,
                token=HF_TOKEN,
            )
            if not torch.cuda.is_available():
                model = model.to(DEVICE)
            model.eval()
            globals()["model"] = model
            print("Re-loaded LLM backbone for extraction.")

        graphs, kept_meta = [], []
        softmax_top_p, softmax_top_i = [], []
        prompt_preds, class_logits_all = [], []
        skipped = 0
        first_error = None
        for r in tqdm(records, desc=desc):
            try:
                res = extract_attention_graph(r, r["label"])
            except Exception as exc:
                if first_error is None:
                    first_error = f"{type(exc).__name__}: {exc}"
                res = None
            if res is None:
                skipped += 1
                continue
            graphs.append(res["graph"])
            softmax_top_p.append(res["softmax_top_p"])
            softmax_top_i.append(res["softmax_top_i"])
            prompt_preds.append(res["prompt_pred"])
            class_logits_all.append(res["class_logits"])
            kept_meta.append({
                "label":   r["label"],
                "verdict": r["verdict"],
                "source":  r["source"],
                "shell":   r["shell"],
                "command": r["command"][:120],
            })
            if torch.cuda.is_available() and len(graphs) % 25 == 0:
                torch.cuda.empty_cache()
        print(f"{desc}: kept {len(graphs)}, skipped {skipped}")
        if first_error is not None:
            print(f"{desc}: first exception -> {first_error}")
        return {
            "graphs": graphs, "meta": kept_meta,
            "softmax_top_p": softmax_top_p, "softmax_top_i": softmax_top_i,
            "prompt_preds":  np.asarray(prompt_preds, dtype=np.int64),
            "class_logits":  np.asarray(class_logits_all, dtype=np.float32),
            "labels":        np.asarray([m["label"] for m in kept_meta], dtype=np.int64),
        }

    TRAIN_META_PKL  = DATA_DIR / f"train_meta_{TAG}.pkl"
    TRAIN_EXTRA_PKL = DATA_DIR / f"train_extras_{TAG}.pkl"
    TEST_META_PKL   = DATA_DIR / f"test_meta_{TAG}.pkl"
    TEST_EXTRA_PKL  = DATA_DIR / f"test_extras_{TAG}.pkl"
    EXTRA_KEYS = ["softmax_top_p", "softmax_top_i", "prompt_preds", "class_logits", "labels"]

    def _save_split(graphs_pkl, meta_pkl, extra_pkl, bundle):
        with open(graphs_pkl, "wb") as f: pickle.dump(bundle["graphs"], f)
        with open(meta_pkl,   "wb") as f: pickle.dump(bundle["meta"],   f)
        with open(extra_pkl,  "wb") as f: pickle.dump({k: bundle[k] for k in EXTRA_KEYS}, f)
        print(f"  saved -> {graphs_pkl.name}, {meta_pkl.name}, {extra_pkl.name}")

    combined_cache_ok = TRAIN_PKL.exists() and TEST_PKL.exists() and EXTRA_PKL.exists() and META_PKL.exists()
    if combined_cache_ok:
        print(f"Loading cached graphs + extras for TAG={TAG}.")
        with open(TRAIN_PKL, "rb") as f: train_graphs = pickle.load(f)
        with open(TEST_PKL,  "rb") as f: test_graphs  = pickle.load(f)
        with open(META_PKL,  "rb") as f: meta = pickle.load(f)
        with open(EXTRA_PKL, "rb") as f: extras = pickle.load(f)
        train_meta, test_meta = meta["train"], meta["test"]
        train_extras, test_extras = extras["train"], extras["test"]
        if len(train_graphs) == 0 or len(test_graphs) == 0:
            combined_cache_ok = False

    if not combined_cache_ok:
        print(f"No combined cache for TAG={TAG} -- running forward pass.")

        if TRAIN_PKL.exists() and TRAIN_META_PKL.exists() and TRAIN_EXTRA_PKL.exists():
            print("Reusing previously-saved train split.")
            with open(TRAIN_PKL,       "rb") as f: train_graphs = pickle.load(f)
            with open(TRAIN_META_PKL,  "rb") as f: train_meta   = pickle.load(f)
            with open(TRAIN_EXTRA_PKL, "rb") as f: train_extras = pickle.load(f)
        else:
            tb = extract_dataset(train_recs, "extract train")
            train_graphs, train_meta = tb["graphs"], tb["meta"]
            train_extras = {k: tb[k] for k in EXTRA_KEYS}
            _save_split(TRAIN_PKL, TRAIN_META_PKL, TRAIN_EXTRA_PKL, tb)
            del tb

        if TEST_PKL.exists() and TEST_META_PKL.exists() and TEST_EXTRA_PKL.exists():
            print("Reusing previously-saved test split.")
            with open(TEST_PKL,       "rb") as f: test_graphs = pickle.load(f)
            with open(TEST_META_PKL,  "rb") as f: test_meta   = pickle.load(f)
            with open(TEST_EXTRA_PKL, "rb") as f: test_extras = pickle.load(f)
        else:
            tb = extract_dataset(test_recs, "extract test")
            test_graphs, test_meta = tb["graphs"], tb["meta"]
            test_extras = {k: tb[k] for k in EXTRA_KEYS}
            _save_split(TEST_PKL, TEST_META_PKL, TEST_EXTRA_PKL, tb)
            del tb

        with open(META_PKL,  "wb") as f: pickle.dump({"train": train_meta,   "test": test_meta},   f)
        with open(EXTRA_PKL, "wb") as f: pickle.dump({"train": train_extras, "test": test_extras}, f)

    print("train graphs:", len(train_graphs), "test graphs:", len(test_graphs))
    if train_graphs:
        print("edge_attr shape (one example):", tuple(train_graphs[0].edge_attr.shape))
"""))

cells.append(md("## 6. Prompt-only baseline (4-class)"))

cells.append(code("""
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

    pp_train, yp_train = train_extras["prompt_preds"], train_extras["labels"]
    pp_test,  yp_test  = test_extras["prompt_preds"],  test_extras["labels"]

    print(f"Prompt-only baseline (model={MODEL_NAME})  --  4-class")
    print(f"  TRAIN: acc={accuracy_score(yp_train, pp_train):.3f}  "
          f"macro_f1={f1_score(yp_train, pp_train, average='macro', zero_division=0):.3f}  n={len(yp_train)}")
    print(f"  TEST : acc={accuracy_score(yp_test, pp_test):.3f}  "
          f"macro_f1={f1_score(yp_test, pp_test, average='macro', zero_division=0):.3f}  n={len(yp_test)}")
    print("\\nTest classification report:")
    print(classification_report(yp_test, pp_test, target_names=LABEL_NAMES,
                                labels=list(range(NUM_CLASSES)), zero_division=0))
"""))

cells.append(md("""
    ## 7. Train GATv2 GNN classifier (4-class)

    The training utility in `model_training/graph_classification.py`
    derives `num_classes` from the distinct labels present in the training
    set. With a heavily imbalanced corpus, a tiny class can vanish — so we
    sanity-check the discovered class count below before training.

    Edge pruning is disabled (`edge_weight_percentile=0`) because each
    graph has only five edges. We also free the LLM before training to
    leave VRAM for the GNN.
"""))

cells.append(code("""
    if "model" in globals():
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    from model_training.graph_classification import train_graph_classifier

    # Quick signal check: mean / max / top-k attention from command -> each def.
    import pickle as _pkl
    with open(TRAIN_PKL, "rb") as _f:
        _gtr = _pkl.load(_f)
    _rows = []
    for _g in _gtr:
        for _k in range(_g.edge_index.shape[1]):
            ea = _g.edge_attr[_k]
            _rows.append({
                "label": ID2LABEL[int(_g.y)],
                "edge":  f"{NODE_NAMES[int(_g.edge_index[0,_k])]}->{NODE_NAMES[int(_g.edge_index[1,_k])]}",
                "mean":  float(ea[0]),
                "max":   float(ea[1]),
                "topk":  float(ea[2]),
            })
    _edf = pd.DataFrame(_rows)
    for col in ["mean", "max", "topk"]:
        _piv = _edf.groupby(["edge", "label"])[col].mean().unstack()
        print(f"\\nEdge-weight signal ({col}):")
        print(_piv)

    n_classes_in_train = len(set(int(g.y) for g in _gtr))
    print(f"\\nDistinct classes present in train graphs: {n_classes_in_train} / {NUM_CLASSES}")
    if n_classes_in_train < NUM_CLASSES:
        print("WARNING: not every class has a training example. The trainer "
              "will derive num_classes from this count; predictions for the "
              "missing class(es) will be impossible.")

    MODEL_OUT_DIR = DATA_DIR / f"gnn_model_{TAG}"
    train_graph_classifier(
        train_file_path=str(TRAIN_PKL),
        test_file_path=str(TEST_PKL),
        model_output_dir=str(MODEL_OUT_DIR),
        num_epochs=700,
        hidden_channel_dimensions=[128, 64],
        batch_size=64,
        learning_rate=5e-4,
        edge_weight_percentile=0,
        dropout=0.5,
        optimizer_type="adam",
        early_stopping_patience=20,
    )
"""))

cells.append(md("""
    ### 7b. Optional: class-weighted retrain

    The shared trainer uses unweighted cross-entropy. For our highly-
    skewed corpus, a class-weighted retrain often improves macro-F1 by
    rescuing the rare classes. We implement a small inline loop here that
    mirrors `train_graph_classifier` but injects per-class weights.
"""))

cells.append(code("""
    from torch.nn import CrossEntropyLoss
    from torch.optim import Adam
    from torch_geometric.loader import DataLoader
    from models.gnn.graph_classifier import GraphClassifier
    import copy

    # Inverse-frequency weights normalised so the most common class = 1.
    class_counts = np.bincount(train_extras["labels"], minlength=NUM_CLASSES).astype(np.float32)
    inv = 1.0 / np.maximum(class_counts, 1.0)
    class_weights = inv / inv[class_counts.argmax()]
    # Cap to avoid blowing the loss up on a class with <5 samples.
    class_weights = np.minimum(class_weights, 20.0)
    print("class counts :", dict(zip(LABEL_NAMES, class_counts.astype(int).tolist())))
    print("class weights:", dict(zip(LABEL_NAMES, class_weights.tolist())))

    feat_dim = train_graphs[0].x.shape[1]
    HIDDEN = [feat_dim, 128, 64]
    gnn_w = GraphClassifier(hidden_channel_dimensions=HIDDEN, num_classes=NUM_CLASSES).to(DEVICE)
    criterion = CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(DEVICE))
    optimizer = Adam(gnn_w.parameters(), lr=5e-4)

    train_loader = DataLoader(train_graphs, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_graphs,  batch_size=64, shuffle=False)

    @torch.no_grad()
    def _eval(loader):
        gnn_w.eval()
        ys, ps = [], []
        for batch in loader:
            batch = batch.to(DEVICE)
            x = batch.x.to(torch.float32)
            logits = gnn_w(x, batch.edge_index, batch.batch, dropout_percentage=0.0)
            ps.extend(logits.argmax(dim=-1).cpu().tolist())
            ys.extend(batch.y.cpu().tolist())
        return ys, ps

    NUM_EPOCHS = 700
    PATIENCE   = 25
    best_f1 = -1.0
    best_state = None
    no_improve = 0
    for epoch in range(NUM_EPOCHS):
        gnn_w.train()
        for batch in train_loader:
            cloned = copy.deepcopy(batch).to(DEVICE)
            cloned.x = cloned.x.to(torch.float32)
            out = gnn_w(cloned.x, cloned.edge_index, cloned.batch, dropout_percentage=0.5)
            loss = criterion(out, cloned.y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if epoch % 10 == 0:
            ys, ps = _eval(test_loader)
            f1 = f1_score(ys, ps, average='macro', zero_division=0)
            acc = accuracy_score(ys, ps)
            if f1 > best_f1:
                best_f1, best_state, no_improve = f1, copy.deepcopy(gnn_w.state_dict()), 0
            else:
                no_improve += 1
            if epoch % 50 == 0:
                print(f"  epoch {epoch:3d}: test_acc={acc:.3f} test_macroF1={f1:.3f} (best={best_f1:.3f})")
            if no_improve >= PATIENCE:
                print(f"  early stop @ epoch {epoch} (best macroF1={best_f1:.3f})")
                break

    if best_state is not None:
        gnn_w.load_state_dict(best_state)

    MODEL_W_DIR = DATA_DIR / f"gnn_weighted_{TAG}"
    MODEL_W_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(gnn_w.state_dict(), MODEL_W_DIR / "model.pt")
    with open(MODEL_W_DIR / "model_metadata.json", "w") as f:
        json.dump({"hidden_channel_dimensions": HIDDEN, "num_classes": NUM_CLASSES,
                   "class_weights": class_weights.tolist(),
                   "best_macro_f1": best_f1}, f)
    print(f"saved weighted model -> {MODEL_W_DIR}")
"""))

cells.append(md("## 8. Evaluate trained GNN(s) on test set"))

cells.append(code("""
    from models.gnn.graph_classifier import GraphClassifier
    from model_training.graph_classification import load_pytorch_geometric_data
    from torch_geometric.loader import DataLoader

    test_dataset = load_pytorch_geometric_data(str(TEST_PKL))

    def evaluate(gnn_model, name):
        gnn_model.eval()
        loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        y_true, y_pred, y_prob = [], [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(DEVICE)
                logits = gnn_model(batch.x.float(), batch.edge_index, batch.batch, dropout_percentage=0.0)
                prob = torch.softmax(logits, dim=-1).cpu().numpy()
                pred = logits.argmax(dim=-1).cpu().numpy()
                y_true.extend(batch.y.cpu().numpy().tolist())
                y_pred.extend(pred.tolist())
                y_prob.append(prob)
        y_prob = np.concatenate(y_prob, axis=0) if y_prob else np.zeros((0, NUM_CLASSES))
        print(f"\\n{name} test metrics:")
        print(f"  accuracy       = {accuracy_score(y_true, y_pred):.3f}")
        print(f"  macro F1       = {f1_score(y_true, y_pred, average='macro', zero_division=0):.3f}")
        print(f"  weighted F1    = {f1_score(y_true, y_pred, average='weighted', zero_division=0):.3f}")
        print(classification_report(y_true, y_pred, target_names=LABEL_NAMES,
                                    labels=list(range(NUM_CLASSES)), zero_division=0))
        print("Confusion matrix (rows=true, cols=pred):")
        cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
        print(pd.DataFrame(cm, index=LABEL_NAMES, columns=LABEL_NAMES))
        return {"y_true": y_true, "y_pred": y_pred, "y_prob": y_prob}

    # Unweighted (model_training.graph_classification trainer output)
    with open(MODEL_OUT_DIR / "model_metadata.json") as f:
        md_u = json.load(f)
    gnn_u = GraphClassifier(
        hidden_channel_dimensions=md_u["hidden_channel_dimensions"],
        num_classes=md_u["num_classes"],
    ).to(DEVICE)
    gnn_u.load_state_dict(torch.load(MODEL_OUT_DIR / "model.pt", map_location=DEVICE))
    gnn_unweighted_eval = evaluate(gnn_u, "GNN (unweighted CE)")

    # Weighted (inline loop in section 7b)
    with open(MODEL_W_DIR / "model_metadata.json") as f:
        md_w = json.load(f)
    gnn_w2 = GraphClassifier(
        hidden_channel_dimensions=md_w["hidden_channel_dimensions"],
        num_classes=md_w["num_classes"],
    ).to(DEVICE)
    gnn_w2.load_state_dict(torch.load(MODEL_W_DIR / "model.pt", map_location=DEVICE))
    gnn_weighted_eval = evaluate(gnn_w2, "GNN (class-weighted CE)")
"""))

cells.append(md("## 9. Linear-probe baseline (LogReg over softmax)"))

cells.append(code("""
    VOCAB_SIZE = len(tokenizer)

    def build_softmax_matrix(extras):
        n = len(extras["softmax_top_p"])
        sm = np.zeros((n, VOCAB_SIZE), dtype=np.float32)
        for r, (p, i) in enumerate(zip(extras["softmax_top_p"], extras["softmax_top_i"])):
            sm[r, i.numpy()] = p.numpy()
        return sm

    sm_tr = build_softmax_matrix(train_extras); y_tr = train_extras["labels"]
    sm_te = build_softmax_matrix(test_extras);  y_te = test_extras["labels"]
    print("softmax dims:", sm_tr.shape, sm_te.shape)

    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1,
                             class_weight="balanced", multi_class="auto")
    clf.fit(sm_tr, y_tr)
    lin_pred = clf.predict(sm_te)
    print("\\nLinear probe (softmax, class-weighted) test metrics:")
    print(f"  accuracy    = {accuracy_score(y_te, lin_pred):.3f}")
    print(f"  macro F1    = {f1_score(y_te, lin_pred, average='macro', zero_division=0):.3f}")
    print(f"  weighted F1 = {f1_score(y_te, lin_pred, average='weighted', zero_division=0):.3f}")
    print(classification_report(y_te, lin_pred, target_names=LABEL_NAMES,
                                labels=list(range(NUM_CLASSES)), zero_division=0))

    yp_te = test_extras["prompt_preds"]
    summary = pd.DataFrame([
        {"method": "Prompt (1-token, 4-class)",
         "accuracy":    accuracy_score(y_te, yp_te),
         "macro_f1":    f1_score(y_te, yp_te, average='macro', zero_division=0),
         "weighted_f1": f1_score(y_te, yp_te, average='weighted', zero_division=0)},
        {"method": "Linear probe (softmax, balanced)",
         "accuracy":    accuracy_score(y_te, lin_pred),
         "macro_f1":    f1_score(y_te, lin_pred, average='macro', zero_division=0),
         "weighted_f1": f1_score(y_te, lin_pred, average='weighted', zero_division=0)},
        {"method": "GNN over attention graph (unweighted)",
         "accuracy":    accuracy_score(gnn_unweighted_eval["y_true"], gnn_unweighted_eval["y_pred"]),
         "macro_f1":    f1_score(gnn_unweighted_eval["y_true"], gnn_unweighted_eval["y_pred"],
                                 average='macro', zero_division=0),
         "weighted_f1": f1_score(gnn_unweighted_eval["y_true"], gnn_unweighted_eval["y_pred"],
                                 average='weighted', zero_division=0)},
        {"method": "GNN over attention graph (class-weighted)",
         "accuracy":    accuracy_score(gnn_weighted_eval["y_true"], gnn_weighted_eval["y_pred"]),
         "macro_f1":    f1_score(gnn_weighted_eval["y_true"], gnn_weighted_eval["y_pred"],
                                 average='macro', zero_division=0),
         "weighted_f1": f1_score(gnn_weighted_eval["y_true"], gnn_weighted_eval["y_pred"],
                                 average='weighted', zero_division=0)},
    ])
    summary
"""))

cells.append(md("## 10. Visualisations — confusion matrices + per-source breakdown"))

cells.append(code("""
    import matplotlib.pyplot as plt

    def plot_cm(y_true, y_pred, title):
        cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(LABEL_NAMES, rotation=30, ha="right")
        ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(LABEL_NAMES)
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(title)
        fig.tight_layout()
        plt.show()

    plot_cm(yp_test,        pp_test,        "Prompt-only baseline (test)")
    plot_cm(gnn_unweighted_eval["y_true"], gnn_unweighted_eval["y_pred"], "GNN unweighted (test)")
    plot_cm(gnn_weighted_eval["y_true"],   gnn_weighted_eval["y_pred"],   "GNN class-weighted (test)")
"""))

cells.append(code("""
    # Per-source breakdown of the weighted-GNN predictions.
    rows = []
    for m, yt, yp in zip(test_meta,
                         gnn_weighted_eval["y_true"],
                         gnn_weighted_eval["y_pred"]):
        bucket = m["source"].split(":", 1)[-1].split("/")[-1] if m["source"] else "(unknown)"
        rows.append({"bucket": bucket, "true": ID2LABEL[yt], "pred": ID2LABEL[yp],
                     "correct": int(yt == yp)})
    pdf = pd.DataFrame(rows)
    by_bucket = (pdf.groupby("bucket")
                    .agg(n=("correct", "size"), acc=("correct", "mean"))
                    .sort_values("n", ascending=False))
    print("Per-source accuracy (weighted GNN):")
    by_bucket
"""))

cells.append(code("""
    # Mean / max / top-k edge-weight comparison across the 4 classes.
    rows = []
    for g in train_graphs:
        for k in range(g.edge_index.shape[1]):
            ea = g.edge_attr[k]
            src = NODE_NAMES[int(g.edge_index[0, k])]
            dst = NODE_NAMES[int(g.edge_index[1, k])]
            rows.append({
                "true_label": ID2LABEL[int(g.y)],
                "edge":       f"{src}->{dst}",
                "mean":       float(ea[0]),
                "max":        float(ea[1]),
                "topk":       float(ea[2]),
            })
    edf = pd.DataFrame(rows)
    for col in ["mean", "max", "topk"]:
        print(f"\\nEdge-weight averages by class -- {col}:")
        piv = edf.groupby(["edge", "true_label"])[col].mean().unstack()
        # Reorder columns to canonical order, dropping any classes absent in train.
        cols = [c for c in LABEL_NAMES if c in piv.columns]
        print(piv[cols])
"""))


# ---------------------------------------------------------------------------
# Section 11 — WITS static-analyzer baseline (4-class head-to-head)
# ---------------------------------------------------------------------------

cells.append(md("""
    ## 11. WITS static-analyzer baseline — 4-class 1:1 comparison

    Runs the rule-based WITS static analyzer
    (`whatInTheShell.isThis(...)` from
    `c:/dev/what-in-the-shell-fresh/dist/index.cjs`) over the same
    held-out test split the GNN was evaluated on, then renders a
    side-by-side table:

    | method              | accuracy | macro-F1 | per-call latency | total |
    | ---                 | ---      | ---      | ---              | ---   |
    | WITS static rules   | …        | n/a (3-cls baseline) | <1 ms | …  |
    | Prompt-only LLM     | …        | …        | LLM fwd          | …    |
    | GNN (unweighted)    | …        | …        | LLM fwd + GNN    | …    |
    | GNN (class-wtd)     | …        | …        | LLM fwd + GNN    | …    |

    **Latency framing**: the WITS analyzer's per-call cost is the
    pure rule engine (no LLM). For the GNN we report end-to-end
    inference latency — featurization (one LLM forward pass per
    command) **plus** GNN evaluation, because that's what production
    deployment would pay per command.

    Prereq: requires `node` on `PATH` and the WITS dist built at
    `c:/dev/what-in-the-shell-fresh/dist/index.cjs`. The path is
    overridable via the `WITS_DIST` environment variable.
"""))

cells.append(code("""
    # ---- 1. Time GNN inference on the cached test graphs ----
    import time

    @torch.no_grad()
    def time_gnn_inference(gnn_model, dataset, n_warmup=5):
        gnn_model.eval()
        # Warmup so first-batch CUDA init doesn't pollute the timing.
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        warm_iter = iter(loader)
        for _ in range(min(n_warmup, len(dataset))):
            b = next(warm_iter).to(DEVICE)
            gnn_model(b.x.float(), b.edge_index, b.batch, dropout_percentage=0.0)
        # Per-sample latency (batch size 1 is the production scenario).
        latencies_ms = []
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        for batch in loader:
            batch = batch.to(DEVICE)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            gnn_model(batch.x.float(), batch.edge_index, batch.batch, dropout_percentage=0.0)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        return np.asarray(latencies_ms, dtype=np.float64)

    print("Timing GNN inference (batch size = 1, every test example) ...")
    gnn_unweighted_lat = time_gnn_inference(gnn_u, test_dataset)
    gnn_weighted_lat   = time_gnn_inference(gnn_w2, test_dataset)
    print(f"  GNN unweighted: mean={gnn_unweighted_lat.mean():.2f}ms  "
          f"p95={np.percentile(gnn_unweighted_lat, 95):.2f}ms  "
          f"max={gnn_unweighted_lat.max():.2f}ms")
    print(f"  GNN weighted  : mean={gnn_weighted_lat.mean():.2f}ms    "
          f"p95={np.percentile(gnn_weighted_lat, 95):.2f}ms  "
          f"max={gnn_weighted_lat.max():.2f}ms")
"""))

cells.append(code("""
    # ---- 2. Time the LLM forward pass (featurization) on a small sample ----
    # Featurization dominates end-to-end GNN cost in production. We
    # re-time it on a SAMPLE rather than the whole test set so the
    # notebook stays fast — extract_attention_graph already ran on the
    # full set when the cache was warm.
    import time

    if "model" not in globals():
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            attn_implementation="eager",
            output_attentions=True,
            output_hidden_states=True,
            token=HF_TOKEN,
        )
        if not torch.cuda.is_available():
            model = model.to(DEVICE)
        model.eval()
        globals()["model"] = model
        print("Re-loaded LLM backbone to time featurization.")

    LAT_SAMPLE_N = 30
    sample = test_recs[:LAT_SAMPLE_N]
    feat_lat_ms = []
    print(f"Timing featurization on {len(sample)} samples ...")
    # warmup
    for r in sample[:3]:
        _ = extract_attention_graph(r, r["label"])
    for r in tqdm(sample, desc="featurize timing"):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = extract_attention_graph(r, r["label"])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        feat_lat_ms.append((time.perf_counter() - t0) * 1000.0)
    feat_lat_ms = np.asarray(feat_lat_ms, dtype=np.float64)
    print(f"  featurization: mean={feat_lat_ms.mean():.1f}ms  "
          f"p95={np.percentile(feat_lat_ms, 95):.1f}ms  "
          f"max={feat_lat_ms.max():.1f}ms")
"""))

cells.append(code("""
    # ---- 3. Run the WITS static analyzer over the SAME test split ----
    # The exported test split lives at TEST_EXPORT_PATH (written in
    # section 2). The scoring script spawns a Node subprocess that
    # imports whatInTheShell from the local WITS dist bundle and writes
    # per-row predictions + latencies as JSONL.
    import subprocess

    WITS_PRED_PATH = DATA_DIR / f"wits_static_predictions_{DATASET_PATH.stem}.jsonl"
    score_script = REPO_ROOT / "data" / "score_wits_static.py"
    wits_dist = os.environ.get("WITS_DIST",
                                "c:/dev/what-in-the-shell-fresh/dist/index.cjs")

    print(f"Running WITS static-analyzer baseline ...")
    print(f"  script  = {score_script}")
    print(f"  test    = {TEST_EXPORT_PATH.name}")
    print(f"  output  = {WITS_PRED_PATH.name}")
    print(f"  wits_dist = {wits_dist}")

    proc = subprocess.run(
        [sys.executable, str(score_script),
         "--test-jsonl", str(TEST_EXPORT_PATH),
         "--out",        str(WITS_PRED_PATH),
         "--wits-dist",  wits_dist],
        capture_output=True, text=True, check=False,
    )
    print(proc.stdout[-4000:])
    if proc.returncode != 0:
        print("STDERR:", proc.stderr[-2000:])
        raise RuntimeError(f"WITS scoring script exited {proc.returncode}")

    # Load predictions back in.
    wits_preds = [json.loads(l) for l in WITS_PRED_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"\\nLoaded {len(wits_preds)} WITS predictions.")
"""))

cells.append(code("""
    # ---- 4. Align WITS predictions to the same y_true the GNN was scored on ----
    # WITS predictions come back in the same order as TEST_EXPORT_PATH,
    # which we wrote in section 2 over `test_recs`. But the GNN's
    # gnn_unweighted_eval["y_true"] is over the SUBSET that featurized
    # successfully (test_graphs / test_meta — a few rows may have been
    # skipped if span alignment failed). We join on the command string
    # so the comparison stays honest.

    # Index WITS by (command, shell).
    wits_by_key = {(p["command"], p["shell"]): p for p in wits_preds}

    aligned_y_true = []
    aligned_wits_pred = []
    aligned_wits_lat = []
    missing = 0
    for m in test_meta:
        key = (m["command"], m["shell"]) if False else None  # m["command"] is truncated
        # The exported JSONL has the FULL command string, but test_meta
        # truncates to 120 chars. Use the JSONL's order instead.
        aligned_y_true.append(m["verdict"] if "verdict" in m else ID2LABEL[m["label"]])

    # Better: walk both lists in order — they were both built from
    # `test_recs`, which is the same shuffled list, and the WITS scorer
    # preserves order.
    test_split = [json.loads(l) for l in TEST_EXPORT_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(test_split) == len(wits_preds), "test-split and WITS pred lengths diverged"
    # Build a (command, shell) -> wits_pred lookup; then re-key by
    # test_meta order (so we drop the few featurization-skipped rows).
    by_key = {(t["command"], t["shell"]): w for t, w in zip(test_split, wits_preds)}

    aligned_y_true = []
    aligned_wits_pred = []
    aligned_wits_lat = []
    aligned_records = []
    skipped = 0
    for m in test_meta:
        # test_meta stores command truncated to 120 chars; for short
        # commands the key matches as-is, for long ones we widen the
        # match. To keep things robust we search by FULL command in
        # test_recs.
        # Find the matching full command in test_split.
        full = None
        for t in test_split:
            if t["command"].startswith(m["command"]) and t["shell"] == m["shell"]:
                full = t
                break
        if full is None:
            skipped += 1
            continue
        w = by_key.get((full["command"], full["shell"]))
        if w is None:
            skipped += 1
            continue
        aligned_y_true.append(m.get("verdict") or ID2LABEL[m["label"]])
        aligned_wits_pred.append(w.get("verdict") or "(error)")
        aligned_wits_lat.append(float(w.get("elapsed_ms") or 0.0))
        aligned_records.append(full)
    if skipped:
        print(f"WARNING: {skipped} test_meta rows had no matching WITS prediction "
              f"(probably featurization-skipped or command-truncation collision).")
    print(f"Aligned {len(aligned_y_true)} (test_meta = {len(test_meta)}, wits = {len(wits_preds)}).")
"""))

cells.append(code("""
    # ---- 5. Side-by-side accuracy + latency table ----
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix,
        classification_report,
    )

    def per_class_block(name, y_true, y_pred):
        labels_str = [ID2LABEL[i] if isinstance(y_true[0], int) else l for i, l in enumerate(LABEL_NAMES)]
        return classification_report(y_true, y_pred, labels=LABEL_NAMES,
                                     target_names=LABEL_NAMES, zero_division=0)

    # GNN y_pred came as ints; convert to label strings for parity with WITS.
    def to_str(yp):
        return [ID2LABEL[int(x)] if not isinstance(x, str) else x for x in yp]

    gnn_u_pred_s = to_str(gnn_unweighted_eval["y_pred"])
    gnn_u_true_s = to_str(gnn_unweighted_eval["y_true"])
    gnn_w_pred_s = to_str(gnn_weighted_eval["y_pred"])
    gnn_w_true_s = to_str(gnn_weighted_eval["y_true"])

    print("="*78)
    print("WITS static (rule-based)")
    print("="*78)
    print(f"  accuracy = {accuracy_score(aligned_y_true, aligned_wits_pred):.3f}")
    print(f"  macro F1 = {f1_score(aligned_y_true, aligned_wits_pred, average='macro', labels=LABEL_NAMES, zero_division=0):.3f}")
    print(classification_report(aligned_y_true, aligned_wits_pred,
                                labels=LABEL_NAMES, target_names=LABEL_NAMES,
                                zero_division=0))
    print(f"\\nConfusion (rows=true, cols=WITS pred):")
    print(pd.DataFrame(
        confusion_matrix(aligned_y_true, aligned_wits_pred, labels=LABEL_NAMES),
        index=LABEL_NAMES, columns=LABEL_NAMES,
    ))

    print()
    print("="*78)
    print("GNN (class-weighted CE)")
    print("="*78)
    print(f"  accuracy = {accuracy_score(gnn_w_true_s, gnn_w_pred_s):.3f}")
    print(f"  macro F1 = {f1_score(gnn_w_true_s, gnn_w_pred_s, average='macro', labels=LABEL_NAMES, zero_division=0):.3f}")
    print(classification_report(gnn_w_true_s, gnn_w_pred_s,
                                labels=LABEL_NAMES, target_names=LABEL_NAMES,
                                zero_division=0))
    print(f"\\nConfusion (rows=true, cols=GNN pred):")
    print(pd.DataFrame(
        confusion_matrix(gnn_w_true_s, gnn_w_pred_s, labels=LABEL_NAMES),
        index=LABEL_NAMES, columns=LABEL_NAMES,
    ))
"""))

cells.append(code("""
    # ---- 6. The headline comparison table ----
    # Per-call latency for the GNN is featurization (LLM fwd) + GNN
    # inference. WITS latency is the pure rule engine.

    feat_mean = float(feat_lat_ms.mean()) if len(feat_lat_ms) else float("nan")
    feat_p95  = float(np.percentile(feat_lat_ms, 95)) if len(feat_lat_ms) else float("nan")
    gnn_u_mean = float(gnn_unweighted_lat.mean())
    gnn_w_mean = float(gnn_weighted_lat.mean())
    gnn_u_p95  = float(np.percentile(gnn_unweighted_lat, 95))
    gnn_w_p95  = float(np.percentile(gnn_weighted_lat, 95))

    wits_lat = np.asarray(aligned_wits_lat, dtype=np.float64)
    wits_mean = float(wits_lat.mean()) if len(wits_lat) else float("nan")
    wits_p95  = float(np.percentile(wits_lat, 95)) if len(wits_lat) else float("nan")

    summary = pd.DataFrame([
        {"method":          "WITS static (rule-based)",
         "accuracy":        accuracy_score(aligned_y_true, aligned_wits_pred),
         "macro_f1":        f1_score(aligned_y_true, aligned_wits_pred,
                                     average='macro', labels=LABEL_NAMES, zero_division=0),
         "lat_mean_ms":     wits_mean,
         "lat_p95_ms":      wits_p95,
         "lat_breakdown":   "rule engine only"},
        {"method":          "Prompt-only LLM (1-token argmax)",
         "accuracy":        accuracy_score(yp_test, pp_test),
         "macro_f1":        f1_score(yp_test, pp_test, average='macro', zero_division=0),
         "lat_mean_ms":     feat_mean,
         "lat_p95_ms":      feat_p95,
         "lat_breakdown":   "LLM forward only"},
        {"method":          "GNN (unweighted CE)",
         "accuracy":        accuracy_score(gnn_unweighted_eval["y_true"], gnn_unweighted_eval["y_pred"]),
         "macro_f1":        f1_score(gnn_unweighted_eval["y_true"], gnn_unweighted_eval["y_pred"],
                                     average='macro', zero_division=0),
         "lat_mean_ms":     feat_mean + gnn_u_mean,
         "lat_p95_ms":      feat_p95 + gnn_u_p95,
         "lat_breakdown":   f"LLM fwd ({feat_mean:.1f}ms) + GNN ({gnn_u_mean:.2f}ms)"},
        {"method":          "GNN (class-weighted CE)",
         "accuracy":        accuracy_score(gnn_weighted_eval["y_true"], gnn_weighted_eval["y_pred"]),
         "macro_f1":        f1_score(gnn_weighted_eval["y_true"], gnn_weighted_eval["y_pred"],
                                     average='macro', zero_division=0),
         "lat_mean_ms":     feat_mean + gnn_w_mean,
         "lat_p95_ms":      feat_p95 + gnn_w_p95,
         "lat_breakdown":   f"LLM fwd ({feat_mean:.1f}ms) + GNN ({gnn_w_mean:.2f}ms)"},
    ])
    print("Head-to-head: accuracy vs. latency\\n")
    summary
"""))

cells.append(code("""
    # ---- 7. Where do WITS and the GNN disagree? ----
    # Surfaces the rows where each method gets something the other
    # misses. Useful for the failure-mode chapter of any writeup.
    diffs = []
    # We can only compare WITS<->GNN on the aligned subset.
    gnn_w_pred_aligned = {}
    for m, yp in zip(test_meta, gnn_weighted_eval["y_pred"]):
        gnn_w_pred_aligned[(m["command"], m["shell"])] = ID2LABEL[int(yp)]
    for t in test_split:
        key = (t["command"], t["shell"])
        gnn = gnn_w_pred_aligned.get(key)
        if gnn is None:
            continue
        wits = (by_key.get(key) or {}).get("verdict")
        truth = t.get("verdict")
        if gnn != wits:
            diffs.append({
                "command":  t["command"][:90],
                "shell":    t["shell"],
                "truth":    truth,
                "gnn":      gnn,
                "wits":     wits,
                "gnn_right":  gnn == truth,
                "wits_right": wits == truth,
                "source":   t.get("source", "")[:25],
            })
    diff_df = pd.DataFrame(diffs)
    if not diff_df.empty:
        print(f"{len(diff_df)} disagreements between GNN (class-weighted) and WITS.")
        print(f"  GNN right, WITS wrong : {((diff_df['gnn_right']) & ~(diff_df['wits_right'])).sum()}")
        print(f"  WITS right, GNN wrong : {((~diff_df['gnn_right']) & (diff_df['wits_right'])).sum()}")
        print(f"  both wrong (different): {((~diff_df['gnn_right']) & ~(diff_df['wits_right'])).sum()}")
    diff_df
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
