# backend/evaluate.py
import csv
import json
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU

# headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .autolabel import build_model, make_preprocessor, compute_prototypes
import torch
from PIL import Image

# --- Paths ---
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LABELED = DATA / "labeled"
EVAL_DIR = DATA / "eval"
OUTPUT = ROOT / "output"
REPORTS = OUTPUT / "reports"
OUTPUT.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

# We now use a direct cosine score threshold instead of confidence
SCORE_THRESHOLD = 0.70  # score must be >= 0.78 to assign a label


# --- Utils ---
def list_eval_images() -> List[Tuple[str, Path]]:
    """Return list of (true_label, path) from data/eval/<class>/*"""
    if not EVAL_DIR.exists():
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    rows = []
    for cls_dir in sorted(EVAL_DIR.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls = cls_dir.name
        for p in sorted(cls_dir.rglob("*")):
            if p.suffix.lower() in exts:
                rows.append((cls, p))
    return rows


def embed_image(model, device, pre, img: Image.Image) -> np.ndarray:
    x = pre(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(x).cpu().numpy().reshape(-1)
    # L2 normalize
    return feat / (np.linalg.norm(feat) + 1e-10)


def save_confusion(cm: np.ndarray,
                   true_labels: List[str],
                   pred_labels: List[str],
                   outpath: Path):
    """Save confusion matrix image with different true/pred label sets."""
    if cm.size == 0:
        return
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xticks(range(len(pred_labels)), pred_labels, rotation=30, ha="right")
    plt.yticks(range(len(true_labels)), true_labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def evaluate():
    # sanity: require eval images
    eval_items = list_eval_images()
    if not eval_items:
        raise RuntimeError("No eval set found at data/eval/<class>/*. Add some held-out images.")

    # model + prototypes from labeled seeds
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(device)
    pre = make_preprocessor()

    classes, prototypes, labeled_feats_stack, labeled_labels_all = compute_prototypes(
        str(LABELED), model, device, min_seeds=1
    )
    if prototypes.shape[0] == 0:
        raise RuntimeError("No prototypes computed from data/labeled.")

    # predict all eval images
    rows_out: List[Dict] = []
    y_true, y_pred = [], []

    for true_label, path in eval_items:
        # read image
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # unreadable -> mark none
            rows_out.append({
                "path": str(path.relative_to(ROOT)),
                "true_label": true_label,
                "predicted_label": "none",
                "score": -1.0,
                "confidence": 0.0,
                "status": "io_error",
            })
            y_true.append(true_label)
            y_pred.append("none")
            continue

        feat = embed_image(model, device, pre, img)
        sims = prototypes.dot(feat)  # cosine vs class prototypes

        # best matching class by cosine similarity
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])

        # confidence just for info/output (not for thresholding)
        confidence = (best_score + 1.0) / 2.0

        # ---------- CUSTOM RULE ----------
        # if cosine score >= SCORE_THRESHOLD -> assign label
        # else -> "none"
        if best_score >= SCORE_THRESHOLD:
            pred = classes[best_idx]
            status = "match"
        else:
            pred = "none"
            status = "no_match"
        # ---------------------------------

        rows_out.append({
            "path": str(path.relative_to(ROOT)),
            "true_label": true_label,
            "predicted_label": pred,
            "score": round(best_score, 6),
            "confidence": round(confidence, 6),
            "status": status,
        })
        y_true.append(true_label)
        y_pred.append(pred)

    # write CSV of eval predictions
    csv_path = OUTPUT / "eval_predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "path", "true_label", "predicted_label", "score", "confidence", "status"
        ])
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    # ---------- Confusion matrix (Option B) ----------
    # True labels: only real classes present in eval set
    true_labels = sorted(set(y_true))

    # Predicted labels: include "none" as a predicted class, but keep it LAST
    pred_labels = sorted(set(y_pred), key=lambda c: (c == "none", c))

    idx_true = {c: i for i, c in enumerate(true_labels)}
    idx_pred = {c: j for j, c in enumerate(pred_labels)}

    cm = np.zeros((len(true_labels), len(pred_labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        i = idx_true[t]
        j = idx_pred[p]
        cm[i, j] += 1
    # -------------------------------------------------

    total = len(rows_out)
    correct = sum(1 for r in rows_out if r["predicted_label"] == r["true_label"])
    acc = correct / total if total else 0.0

    # per-class precision/recall for REAL classes only (no "none")
    per_class = {}
    for c in true_labels:
        i = idx_true[c]
        if c in idx_pred:
            j = idx_pred[c]
            tp = cm[i, j]
            fp = cm[:, j].sum() - tp
        else:
            j = None
            tp = 0
            fp = 0
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        per_class[c] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "support": int(cm[i, :].sum())
        }

    summary = {
        "total_eval_images": total,
        "accuracy": round(acc, 4),
        "per_class": per_class,
        "labels_true": true_labels,
        "labels_pred": pred_labels,
        "csv": csv_path.name,
        "score_threshold": SCORE_THRESHOLD,
    }
    (REPORTS / "overview.json").write_text(json.dumps(summary, indent=2))

    # plots
    save_confusion(cm, true_labels, pred_labels, REPORTS / "confusion_matrix.png")

    print("Reports written to:", REPORTS.resolve())
    print("Eval predictions CSV:", csv_path.resolve())


if __name__ == "__main__":
    evaluate()
