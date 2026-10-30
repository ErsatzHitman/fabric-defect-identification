"""Compute comparison numbers and figures for the report from the two models' prediction files.

Inputs:  cnn/runs/resnet18/{metrics.json,test_predictions.csv}, roboflow_test490.csv,
         predictions.csv (Roboflow, 50 images), cnn_predictions.csv (CNN, 50 images),
         report/roboflow_training.json, fabric_handoff/labels.csv (fabric of batch-1 frames)
Outputs: report/summary.json and report/fig_*.png
"""
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "report"
CLASSES = ["defect_free", "defective"]


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def stem(name):
    """Roboflow export name -> original frame name, e.g. Dimage1__2026..._jpg.rf.<hash>.jpg -> image1__2026....jpg"""
    base = re.sub(r"_jpg\.rf\..*$", ".jpg", name)
    return re.sub(r"^(nd|d|w)", "", base, flags=re.I)


def scores(rows):
    cm = {(t, p): 0 for t in CLASSES for p in CLASSES}
    for r in rows:
        cm[(r["true_label"], r["predicted_label"])] += 1
    n = len(rows)
    acc = sum(cm[(c, c)] for c in CLASSES) / n
    out = {"n": n, "accuracy": acc, "cm": [[cm[(t, p)] for p in CLASSES] for t in CLASSES]}
    for c in CLASSES:
        tp = cm[(c, c)]
        prec = tp / max(1, sum(cm[(t, c)] for t in CLASSES))
        rec = tp / max(1, sum(cm[(c, p)] for p in CLASSES))
        out[f"precision_{c}"], out[f"recall_{c}"] = prec, rec
        out[f"f1_{c}"] = 2 * prec * rec / max(1e-9, prec + rec)
    return out


def latency(rows):
    v = sorted(float(r["latency_ms"]) for r in rows)
    return {"mean_ms": sum(v) / len(v), "median_ms": v[len(v) // 2], "p95_ms": v[int(0.95 * (len(v) - 1))]}


def main():
    fabric = {r["filename"]: r["fabric"] for r in read(ROOT / "fabric_handoff" / "labels.csv")}
    cnn_metrics = json.loads((ROOT / "cnn/runs/resnet18/metrics.json").read_text())
    rf_train = json.loads((REP / "roboflow_training.json").read_text())

    cnn490 = read(ROOT / "cnn/runs/resnet18/test_predictions.csv")
    rf490 = read(ROOT / "roboflow_test490.csv")
    rf50, cnn50 = read(ROOT / "predictions.csv"), read(ROOT / "cnn_predictions.csv")

    summary = {
        "roboflow": {"test490": scores(rf490), "test50": scores(rf50), "latency50": latency(rf50),
                     "latency490": latency(rf490), "training": rf_train},
        "cnn": {"test490": scores(cnn490), "test50": scores(cnn50), "latency50": latency(cnn50),
                "train_seconds": cnn_metrics["train_seconds"], "epochs": cnn_metrics["epochs"],
                "gpu_forward_ms": cnn_metrics["latency_ms_per_image_gpu"], "best_val_acc": cnn_metrics["best_val_acc"],
                "history": cnn_metrics["history"]},
    }

    # Errors on the 490-image test split, with fabric where known (batch-1 frames only).
    rf_by = {r["image"]: r for r in rf490}
    errors = []
    for r in cnn490:
        q = rf_by.get(r["image"])
        cnn_wrong = r["predicted_label"] != r["true_label"]
        rf_wrong = q is not None and q["predicted_label"] != q["true_label"]
        if cnn_wrong or rf_wrong:
            errors.append({"image": r["image"], "true": r["true_label"],
                           "cnn": f'{r["predicted_label"]} ({float(r["confidence"]):.2f})',
                           "roboflow": f'{q["predicted_label"]} ({float(q["confidence"]):.2f})' if q else "",
                           "both_wrong": cnn_wrong and rf_wrong, "cnn_wrong": cnn_wrong, "rf_wrong": rf_wrong,
                           "fabric": fabric.get(stem(r["image"]), "batch 2 (unknown)")})
    summary["errors"] = errors
    by = {}
    for e in errors:
        for m in ("cnn", "rf"):
            if e[f"{m}_wrong"]:
                k = (m, e["true"])
                by[k] = by.get(k, 0) + 1
    summary["error_counts"] = {f"{m}:{t}": v for (m, t), v in by.items()}
    (REP / "summary.json").write_text(json.dumps(summary, indent=2))

    # Figure 1: confusion matrices on the 490-image test split.
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    for ax, (title, s) in zip(axes, [("Roboflow ViT-B/16", summary["roboflow"]["test490"]),
                                     ("ResNet18 (PyTorch)", summary["cnn"]["test490"])]):
        cm = s["cm"]
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i][j], ha="center", va="center",
                        color="white" if cm[i][j] > max(map(max, cm)) / 2 else "black", fontsize=12)
        ax.set_xticks([0, 1], CLASSES)
        ax.set_yticks([0, 1], CLASSES)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{title}\naccuracy {s['accuracy']:.1%} (n={s['n']})", fontsize=10)
    fig.tight_layout()
    fig.savefig(REP / "fig_confusion.png", dpi=200)

    # Figure 2: a sheet of misclassified test images (both-wrong first).
    # Show different failure types: up to 2 per (fabric, true class) group, most confident errors first.
    errors.sort(key=lambda e: (not e["both_wrong"], e["true"]))
    groups = {}
    for e in errors:
        groups.setdefault((e["fabric"], e["true"]), []).append(e)
    show = []
    for _ in range(2):
        for g in sorted(groups):
            if groups[g] and len(show) < 8:
                show.append(groups[g].pop(0))
    if show:
        cols = 4
        rows_n = (len(show) + cols - 1) // cols
        fig, axes = plt.subplots(rows_n, cols, figsize=(8, 2.3 * rows_n))
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
        for ax in axes:
            ax.axis("off")
        for ax, e in zip(axes, show):
            p = ROOT / "data/v1/test" / e["true"] / e["image"]
            ax.imshow(Image.open(p))
            ax.set_title(f"{e['fabric'].replace(' (unknown)', '')} | true: {e['true']}\n"
                         f"RF: {e['roboflow']}\nCNN: {e['cnn']}", fontsize=6.5)
        fig.tight_layout()
        fig.savefig(REP / "fig_errors.png", dpi=200)
    print(json.dumps({k: v for k, v in summary.items() if k != "errors"}, indent=1)[:4000])
    print(len(errors), "test images misclassified by at least one model")


if __name__ == "__main__":
    main()
