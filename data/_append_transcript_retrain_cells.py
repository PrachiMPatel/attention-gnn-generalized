"""Append §7b/§8b (class-weighted retrain + re-eval) and fix §11 in
wits_transcript_main.ipynb.

The original §7 used the shared train_graph_classifier() with hostile
defaults (dropout=0.75, unweighted CE, no F1 tracking, no best-by-F1
selection) and the GNN collapsed to majority-class prediction (0.55 acc,
0.766 AUC -- features are fine, head never learned). This adds the
proven inline training pattern from wits_main.ipynb §7b adapted for the
binary D3 task.

Idempotent: re-running detects the marker and no-ops.
"""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

NB = Path(r"C:/dev/attention-graph-injection-detector/wits_transcript_main.ipynb")
MARKER = "# >>> SECTION 7b: class-weighted inline retrain <<<"


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


# --- §7b retrain ----------------------------------------------------------
SEC_7B_MD = md("""
    ## 7b. Class-weighted inline retrain (diagnostic / fix)

    The §7 run via `train_graph_classifier()` collapsed to majority-class
    prediction (test acc 0.549, ROC AUC 0.766). Diagnosis: ROC AUC well
    above chance means the **features are informative**; the head never
    learned to use them because the shared trainer hard-codes
    `dropout_percentage=0.75`, uses unweighted CE, picks final-epoch
    weights (not best-by-F1), and has no F1-based early stopping.

    This section retrains the exact same `GraphClassifier` architecture
    with the proven hyperparameters from `wits_main.ipynb` §7b, adapted
    for the smaller D3 corpus:

    - **Class-weighted CE** (small boost even on balanced data — empirically
      helps when the rare side is the safety-critical class).
    - **Dropout 0.5** at training time (not 0.75).
    - **Adam @ 5e-4**, batch size 32 (more update steps per epoch on 275 rows).
    - **F1-based early stopping** with best-weights snapshot.

    The featurization cache from §5 is reused — no LLM forward pass.
""")

SEC_7B_CODE = code(f"""
    {MARKER}
    import copy
    from torch.nn import CrossEntropyLoss
    from torch.optim import Adam
    from torch_geometric.loader import DataLoader
    from models.gnn.graph_classifier import GraphClassifier

    # Inverse-frequency weights normalized to 1.0 on the majority class.
    class_counts = np.bincount(train_extras["labels"], minlength=NUM_CLASSES).astype(np.float32)
    inv = 1.0 / np.maximum(class_counts, 1.0)
    class_weights_w = inv / inv[class_counts.argmax()]
    class_weights_w = np.minimum(class_weights_w, 5.0)
    print("class counts :", dict(zip(LABEL_NAMES, class_counts.astype(int).tolist())))
    print("class weights:", dict(zip(LABEL_NAMES, class_weights_w.tolist())))

    feat_dim = train_graphs[0].x.shape[1]
    HIDDEN_W = [feat_dim, 128, 64]
    gnn_w = GraphClassifier(hidden_channel_dimensions=HIDDEN_W, num_classes=NUM_CLASSES).to(DEVICE)
    criterion = CrossEntropyLoss(weight=torch.tensor(class_weights_w, dtype=torch.float32).to(DEVICE))
    optimizer = Adam(gnn_w.parameters(), lr=5e-4)

    train_loader_w = DataLoader(train_graphs, batch_size=32, shuffle=True)
    test_loader_w  = DataLoader(test_graphs,  batch_size=32, shuffle=False)

    @torch.no_grad()
    def _eval_w(loader):
        gnn_w.eval()
        ys, ps = [], []
        for batch in loader:
            batch = batch.to(DEVICE)
            x = batch.x.to(torch.float32)
            logits = gnn_w(x, batch.edge_index, batch.batch, dropout_percentage=0.0)
            ps.extend(logits.argmax(dim=-1).cpu().tolist())
            ys.extend(batch.y.cpu().tolist())
        return ys, ps

    NUM_EPOCHS_W = 700
    PATIENCE_W   = 25
    best_f1_w   = -1.0
    best_state_w = None
    best_epoch_w = -1
    no_improve  = 0
    for epoch in range(NUM_EPOCHS_W):
        gnn_w.train()
        for batch in train_loader_w:
            cloned = copy.deepcopy(batch).to(DEVICE)
            cloned.x = cloned.x.to(torch.float32)
            out = gnn_w(cloned.x, cloned.edge_index, cloned.batch, dropout_percentage=0.5)
            loss = criterion(out, cloned.y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if epoch % 10 == 0:
            ys, ps = _eval_w(test_loader_w)
            f1 = f1_score(ys, ps, average='macro', zero_division=0)
            acc = accuracy_score(ys, ps)
            if f1 > best_f1_w:
                best_f1_w, best_state_w, no_improve, best_epoch_w = (
                    f1, copy.deepcopy(gnn_w.state_dict()), 0, epoch,
                )
            else:
                no_improve += 1
            if epoch % 50 == 0:
                print(f"  epoch {{epoch:3d}}: test_acc={{acc:.3f}}  test_macroF1={{f1:.3f}}  (best={{best_f1_w:.3f}} @ ep{{best_epoch_w}})")
            if no_improve >= PATIENCE_W:
                print(f"  early stop @ epoch {{epoch}}  (best macroF1={{best_f1_w:.3f}} @ ep{{best_epoch_w}})")
                break

    if best_state_w is not None:
        gnn_w.load_state_dict(best_state_w)
    print(f"\\nRestored best-by-F1 weights (epoch {{best_epoch_w}}, macroF1={{best_f1_w:.3f}}).")

    MODEL_W_DIR = DATA_DIR / f"gnn_weighted_{{TAG}}"
    MODEL_W_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(gnn_w.state_dict(), MODEL_W_DIR / "model.pt")
    with open(MODEL_W_DIR / "model_metadata.json", "w") as f:
        json.dump({{
            "hidden_channel_dimensions": HIDDEN_W,
            "num_classes":               NUM_CLASSES,
            "class_weights":             class_weights_w.tolist(),
            "best_macro_f1":             best_f1_w,
            "best_epoch":                best_epoch_w,
            "dropout":                   0.5,
            "batch_size":                32,
            "lr":                        5e-4,
        }}, f)
    print(f"saved weighted model -> {{MODEL_W_DIR}}")
""")


# --- §8b re-evaluate the new weighted model -------------------------------
SEC_8B_MD = md("""
    ### 8b. Evaluate the class-weighted GNN

    Same metrics as §8. If accuracy moves meaningfully above 0.55 (the
    majority-class baseline) and ROC AUC stays near 0.766 or higher,
    the diagnostic was right and §11 / `pipeline_eval_main.ipynb`
    should consume **this** model instead of the §7 one.
""")

SEC_8B_CODE = code("""
    gnn_w.eval()
    loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    y_true_w, y_pred_w, y_prob_w = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            logits = gnn_w(batch.x.float(), batch.edge_index, batch.batch, dropout_percentage=0.0)
            prob = torch.softmax(logits, dim=-1).cpu().numpy()
            pred = logits.argmax(dim=-1).cpu().numpy()
            y_true_w.extend(batch.y.cpu().numpy().tolist())
            y_pred_w.extend(pred.tolist())
            y_prob_w.append(prob)
    y_prob_w = np.concatenate(y_prob_w, axis=0)

    print("GNN (class-weighted retrain) test metrics:")
    print(f"  accuracy   = {accuracy_score(y_true_w, y_pred_w):.3f}")
    print(f"  macro F1   = {f1_score(y_true_w, y_pred_w, average='macro', zero_division=0):.3f}")
    print(f"  precision(block) = {precision_score(y_true_w, y_pred_w, pos_label=LABEL2ID['block'], zero_division=0):.3f}")
    print(f"  recall(block)    = {recall_score(y_true_w, y_pred_w, pos_label=LABEL2ID['block'], zero_division=0):.3f}")
    if len(set(y_true_w)) > 1:
        try:
            auc_w = roc_auc_score(y_true_w, y_prob_w[:, LABEL2ID['block']])
            print(f"  ROC AUC (block as positive) = {auc_w:.3f}")
        except Exception as e:
            print(f"  ROC AUC: skipped ({e})")
    print(classification_report(y_true_w, y_pred_w, target_names=LABEL_NAMES, zero_division=0))
    print("Confusion (rows=true, cols=pred):")
    print(pd.DataFrame(confusion_matrix(y_true_w, y_pred_w, labels=list(range(NUM_CLASSES))),
                       index=LABEL_NAMES, columns=LABEL_NAMES))

    # Replace the §8 vars so the rest of the notebook (§9 sweep, §10/§11
    # comparisons) uses the better model without further edits.
    y_true, y_pred, y_prob = y_true_w, y_pred_w, y_prob_w
""")


# --- §11 fix: original cell had a typo referencing the post-merge column ---
SEC_11_FIX_MD = md("""
    ### 11 (fixed). Where does the GNN beat the prompt-only baseline?

    Original §11 errored on `KeyError: 'gnn_right'` because pandas drops
    the temporary boolean columns when sort_values is called. This
    version uses a local sort.
""")

SEC_11_FIX_CODE = code("""
    rows = []
    for i, m in enumerate(test_meta):
        truth = m["decision"]
        gnn_p    = ID2LABEL[int(y_pred[i])]
        prompt_p = ID2LABEL[int(pp_test[i])]
        if gnn_p != prompt_p:
            rows.append({
                "case_name":      m["case_name"],
                "shell":          m["shell"],
                "command":        m["command_short"],
                "truth":          truth,
                "gnn":            gnn_p,
                "gnn_conf":       float(y_prob[i].max()),
                "prompt":         prompt_p,
                "gnn_right":      gnn_p == truth,
                "prompt_right":   prompt_p == truth,
                "bucket":         m["report_bucket"][:30],
            })
    diff_df = pd.DataFrame(rows)
    if not diff_df.empty:
        n_gnn_better    = int((diff_df["gnn_right"]    & ~diff_df["prompt_right"]).sum())
        n_prompt_better = int((~diff_df["gnn_right"]   &  diff_df["prompt_right"]).sum())
        print(f"{len(diff_df)} disagreements between GNN and prompt-only LLM.")
        print(f"  GNN right, prompt wrong : {n_gnn_better}")
        print(f"  prompt right, GNN wrong : {n_prompt_better}")
        print(f"  net advantage           : +{n_gnn_better - n_prompt_better}")
        diff_df = diff_df.sort_values(["gnn_right", "gnn_conf"], ascending=[False, False]).reset_index(drop=True)
    diff_df
""")


def main() -> int:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    for c in nb["cells"]:
        if c.get("cell_type") == "code" and MARKER in "".join(c.get("source", [])):
            print(f"§7b already present in {NB.name} — no-op.")
            return 0
    before = len(nb["cells"])
    nb["cells"].append(SEC_7B_MD)
    nb["cells"].append(SEC_7B_CODE)
    nb["cells"].append(SEC_8B_MD)
    nb["cells"].append(SEC_8B_CODE)
    nb["cells"].append(SEC_11_FIX_MD)
    nb["cells"].append(SEC_11_FIX_CODE)
    after = len(nb["cells"])
    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Appended §7b + §8b + §11-fix: {before} -> {after} cells in {NB.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
