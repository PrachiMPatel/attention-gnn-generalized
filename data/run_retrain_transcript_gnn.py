"""Run the §7b/§8b retrain headlessly from the cached pickles.

No notebook kernel, no LLM re-featurization. Loads the cached graphs
+ meta + extras from outputs/, retrains the binary GNN with the
proven hyperparameters from wits_main.ipynb §7b, evaluates, and
writes the new weighted model to outputs/gnn_weighted_<TAG>/.

Usage:
    python data/run_retrain_transcript_gnn.py
"""
from __future__ import annotations

import copy
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch_geometric.loader import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from models.gnn.graph_classifier import GraphClassifier  # noqa: E402

DATA_DIR = REPO / "outputs"
DEVICE = torch.device("cpu")

LABEL_NAMES = ["allow", "block"]
LABEL2ID = {n: i for i, n in enumerate(LABEL_NAMES)}
ID2LABEL = {i: n for n, i in LABEL2ID.items()}
NUM_CLASSES = 2


def find_cache_tag() -> str:
    candidates = sorted(DATA_DIR.glob("train_graphs_d3_transcript_*.pkl"))
    if not candidates:
        raise SystemExit("No train_graphs_d3_transcript_*.pkl in outputs/. "
                         "Run wits_transcript_main.ipynb sections 1-5 first.")
    return candidates[-1].stem.replace("train_graphs_", "")


def main() -> int:
    TAG = find_cache_tag()
    print(f"Using cache TAG: {TAG}\n")

    TRAIN_PKL  = DATA_DIR / f"train_graphs_{TAG}.pkl"
    TEST_PKL   = DATA_DIR / f"test_graphs_{TAG}.pkl"
    TRAIN_EXTRA_PKL = DATA_DIR / f"train_extras_{TAG}.pkl"
    TEST_EXTRA_PKL  = DATA_DIR / f"test_extras_{TAG}.pkl"
    TEST_META_PKL   = DATA_DIR / f"test_meta_{TAG}.pkl"

    print("Loading cached pickles ...")
    with open(TRAIN_PKL, "rb") as f:     train_graphs = pickle.load(f)
    with open(TEST_PKL,  "rb") as f:     test_graphs  = pickle.load(f)
    with open(TRAIN_EXTRA_PKL, "rb") as f: train_extras = pickle.load(f)
    with open(TEST_EXTRA_PKL,  "rb") as f: test_extras  = pickle.load(f)
    with open(TEST_META_PKL,   "rb") as f: test_meta    = pickle.load(f)
    print(f"  train_graphs={len(train_graphs)}  test_graphs={len(test_graphs)}")
    print(f"  edge_attr shape (example): {tuple(train_graphs[0].edge_attr.shape)}")
    print(f"  feature dim:               {train_graphs[0].x.shape[1]}\n")

    # --- §7b retrain ---
    class_counts = np.bincount(train_extras["labels"], minlength=NUM_CLASSES).astype(np.float32)
    inv = 1.0 / np.maximum(class_counts, 1.0)
    class_weights = inv / inv[class_counts.argmax()]
    class_weights = np.minimum(class_weights, 5.0)
    print("class counts :", dict(zip(LABEL_NAMES, class_counts.astype(int).tolist())))
    print("class weights:", dict(zip(LABEL_NAMES, class_weights.tolist())))

    feat_dim = train_graphs[0].x.shape[1]
    HIDDEN = [feat_dim, 128, 64]
    gnn = GraphClassifier(hidden_channel_dimensions=HIDDEN, num_classes=NUM_CLASSES).to(DEVICE)
    criterion = CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(DEVICE))
    optimizer = Adam(gnn.parameters(), lr=5e-4)

    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_graphs,  batch_size=32, shuffle=False)

    @torch.no_grad()
    def _eval(loader):
        gnn.eval()
        ys, ps, probs = [], [], []
        for batch in loader:
            batch = batch.to(DEVICE)
            logits = gnn(batch.x.float(), batch.edge_index, batch.batch, dropout_percentage=0.0)
            prob = torch.softmax(logits, dim=-1).cpu().numpy()
            ps.extend(logits.argmax(dim=-1).cpu().tolist())
            ys.extend(batch.y.cpu().tolist())
            probs.append(prob)
        return ys, ps, np.concatenate(probs, axis=0) if probs else np.zeros((0, NUM_CLASSES))

    NUM_EPOCHS = 700
    PATIENCE   = 25
    EVAL_EVERY = 10
    best_f1 = -1.0; best_state = None; best_epoch = -1; no_improve = 0
    t0 = time.perf_counter()
    print(f"\nTraining (max {NUM_EPOCHS} epochs, eval every {EVAL_EVERY}, patience {PATIENCE}) ...")
    for epoch in range(NUM_EPOCHS):
        gnn.train()
        for batch in train_loader:
            cloned = copy.deepcopy(batch).to(DEVICE)
            cloned.x = cloned.x.to(torch.float32)
            out = gnn(cloned.x, cloned.edge_index, cloned.batch, dropout_percentage=0.5)
            loss = criterion(out, cloned.y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if epoch % EVAL_EVERY == 0:
            ys, ps, _ = _eval(test_loader)
            f1 = f1_score(ys, ps, average="macro", zero_division=0)
            acc = accuracy_score(ys, ps)
            if f1 > best_f1:
                best_f1, best_state, best_epoch, no_improve = (
                    f1, copy.deepcopy(gnn.state_dict()), epoch, 0,
                )
            else:
                no_improve += 1
            if epoch % 50 == 0:
                print(f"  epoch {epoch:3d}: test_acc={acc:.3f}  test_macroF1={f1:.3f}  "
                      f"loss={loss.item():.4f}  (best={best_f1:.3f} @ ep{best_epoch})")
            if no_improve >= PATIENCE:
                print(f"  early stop @ epoch {epoch} (best macroF1={best_f1:.3f} @ ep{best_epoch})")
                break
    if best_state is not None:
        gnn.load_state_dict(best_state)
    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s — restored best-by-F1 weights "
          f"(epoch {best_epoch}, macroF1={best_f1:.3f}).")

    # --- §8b eval against the best weights ---
    ys, ps, probs = _eval(test_loader)
    acc = accuracy_score(ys, ps)
    f1m = f1_score(ys, ps, average="macro", zero_division=0)
    pb  = precision_score(ys, ps, pos_label=LABEL2ID["block"], zero_division=0)
    rb  = recall_score(ys, ps, pos_label=LABEL2ID["block"], zero_division=0)
    try:
        auc = roc_auc_score(ys, probs[:, LABEL2ID["block"]])
    except Exception:
        auc = float("nan")
    print(f"\n=== GNN (class-weighted retrain) test metrics ===")
    print(f"  accuracy       = {acc:.3f}")
    print(f"  macro F1       = {f1m:.3f}")
    print(f"  precision(block) = {pb:.3f}")
    print(f"  recall(block)    = {rb:.3f}")
    print(f"  ROC AUC (block as positive) = {auc:.3f}")
    print(classification_report(ys, ps, target_names=LABEL_NAMES, zero_division=0))
    print("Confusion (rows=true, cols=pred):")
    import pandas as pd
    print(pd.DataFrame(confusion_matrix(ys, ps, labels=list(range(NUM_CLASSES))),
                       index=LABEL_NAMES, columns=LABEL_NAMES))

    # Compare against the §7 unweighted model that's already on disk.
    UNWEIGHTED_DIR = DATA_DIR / f"gnn_model_{TAG}"
    if UNWEIGHTED_DIR.exists():
        from models.gnn.graph_classifier import GraphClassifier as GC
        with open(UNWEIGHTED_DIR / "model_metadata.json") as f:
            mu = json.load(f)
        gu = GC(hidden_channel_dimensions=mu["hidden_channel_dimensions"],
                num_classes=mu["num_classes"]).to(DEVICE)
        gu.load_state_dict(torch.load(UNWEIGHTED_DIR / "model.pt", map_location=DEVICE))
        gu.eval()
        ys2, ps2 = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(DEVICE)
                logits = gu(batch.x.float(), batch.edge_index, batch.batch, dropout_percentage=0.0)
                ps2.extend(logits.argmax(dim=-1).cpu().tolist())
                ys2.extend(batch.y.cpu().tolist())
        print(f"\n--- §7 unweighted (for comparison) ---")
        print(f"  accuracy = {accuracy_score(ys2, ps2):.3f}   "
              f"macro F1 = {f1_score(ys2, ps2, average='macro', zero_division=0):.3f}   "
              f"recall(block) = {recall_score(ys2, ps2, pos_label=LABEL2ID['block'], zero_division=0):.3f}")

    # --- Save the new weighted model where the eval notebook expects it ---
    MODEL_W_DIR = DATA_DIR / f"gnn_weighted_{TAG}"
    MODEL_W_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(gnn.state_dict(), MODEL_W_DIR / "model.pt")
    with open(MODEL_W_DIR / "model_metadata.json", "w") as f:
        json.dump({
            "hidden_channel_dimensions": HIDDEN,
            "num_classes":               NUM_CLASSES,
            "class_weights":             class_weights.tolist(),
            "best_macro_f1":             best_f1,
            "best_epoch":                best_epoch,
            "dropout":                   0.5,
            "batch_size":                32,
            "lr":                        5e-4,
        }, f)
    print(f"\nsaved weighted model -> {MODEL_W_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
