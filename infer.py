"""Run the trained Roboflow classification model on unseen test images and export predictions.csv.

Usage:
    python infer.py --images test_images --version 1

Reads ROBOFLOW_API_KEY from .env (never commit it). If --images contains one sub-folder per class
(e.g. test_images/defective/*.jpg), the true label is also written and accuracy is printed.
Each call is timed, so predictions.csv also carries the per-image API latency.
"""
import argparse
import base64
import csv
import os
import statistics
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

WORKSPACE = "nirupama-sinha"
PROJECT = "fabric-defect-identification"
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def predict(session, url, path):
    payload = base64.b64encode(path.read_bytes())
    t = time.perf_counter()
    resp = session.post(url, data=payload, headers={"Content-Type": "application/x-www-form-urlencoded"},
                        timeout=60)
    latency_ms = (time.perf_counter() - t) * 1000
    resp.raise_for_status()
    body = resp.json()
    # Single-label classification response: {"top": <class>, "confidence": <float>, "predictions": [...]}
    if "top" in body:
        return body["top"], float(body["confidence"]), latency_ms
    best = max(body["predictions"], key=lambda p: p["confidence"])
    return best["class"], float(best["confidence"]), latency_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, default=Path("test_images"))
    ap.add_argument("--version", type=int, required=True, help="Roboflow model version number")
    ap.add_argument("--out", type=Path, default=Path("predictions.csv"))
    args = ap.parse_args()

    load_dotenv()
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise SystemExit("Set ROBOFLOW_API_KEY in .env")
    url = f"https://classify.roboflow.com/{PROJECT}/{args.version}?api_key={key}"

    files = sorted(p for p in args.images.rglob("*") if p.suffix.lower() in EXTS)
    if not files:
        raise SystemExit(f"No images found in {args.images}")

    rows, latencies, correct, labelled = [], [], 0, 0
    with requests.Session() as session:
        for i, path in enumerate(files, 1):
            label, conf, ms = predict(session, url, path)
            true = path.parent.name if path.parent != args.images else ""
            rows.append({"image": path.name, "predicted_label": label, "confidence": f"{conf:.4f}",
                         "true_label": true, "latency_ms": f"{ms:.1f}"})
            latencies.append(ms)
            if true:
                labelled += 1
                correct += label == true
            print(f"[{i}/{len(files)}] {path.name}: {label} ({conf:.3f}) {ms:.0f} ms")

    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "predicted_label", "confidence", "true_label", "latency_ms"])
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {args.out}")
    print(f"Latency per image: mean {statistics.mean(latencies):.0f} ms, "
          f"median {statistics.median(latencies):.0f} ms")
    if labelled:
        print(f"Accuracy on labelled images: {correct}/{labelled} = {correct / labelled:.1%}")


if __name__ == "__main__":
    main()
