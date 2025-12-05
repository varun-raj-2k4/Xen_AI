# backend/plot_eval.py
import csv
import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np

# use a non-GUI backend so it runs fine on servers/CI/terminal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Paths ---
ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"
REPORTS_DIR = OUTPUT_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Helpers ---
def find_latest_csv(output_dir: Path) -> Path:
    csvs = sorted(output_dir.glob("*.csv"),
                  key=lambda p: p.stat().st_mtime,
                  reverse=True)
    if not csvs:
        raise FileNotFoundError(f"No CSVs in {output_dir}. Run autolabel first.")
    return csvs[0]

def read_results(csv_path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            assigned = r.get("assigned_class")
            if assigned in (None, "", "None", "none"):
                assigned = None
            score_str = r.get("score", "0")
            try:
                score = float(score_str)
            except Exception:
                score = 0.0
            rec = {
                "path": r.get("path", ""),
                "assigned_class": assigned,
                "score": score,
                "status": r.get("status", ""),
            }
            # optional GT column if user added it to CSV
            if "true_label" in r and r["true_label"] != "":
                rec["true_label"] = r["true_label"]
            rows.append(rec)
    return rows

def per_class_stats(rows: List[Dict]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for r in rows:
        cls = r["assigned_class"]
        if cls is None:
            continue
        d = stats.setdefault(cls, {"count": 0, "score_sum": 0.0})
        d["count"] += 1
        d["score_sum"] += r["score"]
    for cls, d in stats.items():
        d["avg_score"] = (d["score_sum"] / d["count"]) if d["count"] > 0 else 0.0
    return stats

def threshold_sweep(rows: List[Dict], thresholds: np.ndarray) -> List[Dict[str, float]]:
    """
    Acceptance (match fraction) vs requested threshold.
    We mirror the API confidence: confidence = (score + 1) / 2, with an effective minimum of 0.75.
    """
    metrics = []
    if not rows:
        return [{"threshold": float(t), "effective_threshold": max(0.75, float(t)), "accept_rate": 0.0}
                for t in thresholds]
    confidences = np.array([(r["score"] + 1.0) / 2.0 for r in rows], dtype=float)
    n = float(len(rows))
    for t in thresholds:
        eff_t = max(0.75, float(t))
        accepted = float((confidences >= eff_t).sum())
        metrics.append({"threshold": float(t),
                        "effective_threshold": eff_t,
                        "accept_rate": accepted / n})
    return metrics

def maybe_confusion(rows: List[Dict]) -> Tuple[np.ndarray, List[str]]:
    """Build confusion matrix if 'true_label' exists for all rows."""
    if not rows or not all(("true_label" in r) for r in rows):
        return np.array([]), []
    y_true = [r["true_label"] for r in rows]
    y_pred = [(r["assigned_class"] or "none") for r in rows]
    labels = sorted(list(set(y_true + y_pred)))
    idx = {c: i for i, c in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1
    return cm, labels

# --- Plots ---
def plot_status_pie(rows: List[Dict], outpath: Path):
    status_counts: Dict[str, int] = {}
    for r in rows:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
    labels = list(status_counts.keys())
    sizes = [status_counts[k] for k in labels]
    plt.figure()
    if sizes:
        plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
    plt.title("Match vs No-match (Status)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

def plot_score_hist(rows: List[Dict], outpath: Path):
    scores = [r["score"] for r in rows]
    plt.figure()
    if scores:
        plt.hist(scores, bins=30)
    plt.xlabel("Cosine similarity score")
    plt.ylabel("Count")
    plt.title("Distribution of Scores")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

def plot_per_class(stats: Dict[str, Dict[str, float]], out_counts: Path, out_avgs: Path):
    if not stats:
        return
    classes = list(stats.keys())
    counts = [stats[c]["count"] for c in classes]
    avgs = [stats[c]["avg_score"] for c in classes]

    # counts
    plt.figure(figsize=(max(6, len(classes)*0.7), 4))
    plt.bar(classes, counts)
    plt.ylabel("Accepted images")
    plt.title("Per-class accepted count")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_counts, dpi=160)
    plt.close()

    # average score
    plt.figure(figsize=(max(6, len(classes)*0.7), 4))
    plt.bar(classes, avgs)
    plt.ylabel("Average score")
    plt.title("Per-class average similarity score")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_avgs, dpi=160)
    plt.close()

def plot_threshold_curve(metrics: List[Dict[str, float]], outpath: Path):
    th = [m["threshold"] for m in metrics]
    acc = [m["accept_rate"] for m in metrics]
    plt.figure()
    if th:
        plt.plot(th, acc)
    plt.xlabel("Requested threshold")
    plt.ylabel("Acceptance rate")
    plt.title("Acceptance vs Threshold (effective min 0.75 applied)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

def plot_confusion(cm: np.ndarray, labels: List[str], outpath: Path):
    if cm.size == 0:
        return
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

# --- Main ---
def main():
    csv_path = find_latest_csv(OUTPUT_DIR)
    rows = read_results(csv_path)

    total = len(rows)
    matches = sum(1 for r in rows if r["status"] == "match")
    no_matches = total - matches

    stats = per_class_stats(rows)
    thresholds = np.linspace(0.5, 0.95, 10)
    sweep = threshold_sweep(rows, thresholds)

    # JSON summaries
    overview = {
        "csv": csv_path.name,
        "total_images": total,
        "matches": matches,
        "no_matches": no_matches,
        "per_class": stats,
    }
    (REPORTS_DIR / "overview.json").write_text(json.dumps(overview, indent=2))
    (REPORTS_DIR / "threshold_metrics.json").write_text(json.dumps(sweep, indent=2))

    # Plots
    plot_status_pie(rows, REPORTS_DIR / "status_pie.png")
    plot_score_hist(rows, REPORTS_DIR / "score_hist.png")
    plot_per_class(stats, REPORTS_DIR / "per_class_counts.png",
                   REPORTS_DIR / "per_class_avg_score.png")
    plot_threshold_curve(sweep, REPORTS_DIR / "acceptance_vs_threshold.png")

    # Optional confusion matrix (only if CSV has true_label)
    cm, labels = maybe_confusion(rows)
    if cm.size:
        plot_confusion(cm, labels, REPORTS_DIR / "confusion_matrix.png")

    print("Reports written to:", REPORTS_DIR.resolve())

if __name__ == "__main__":
    main()
