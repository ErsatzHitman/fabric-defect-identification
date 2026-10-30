"""Run the fine-tuned ResNet18 on the same 50 test images as infer.py -> cnn_predictions.csv.

    python cnn/predict_cnn.py --images test_images --weights cnn/runs/resnet18/best.pt
"""
import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
EXTS = {".jpg", ".jpeg", ".png"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, default=Path("test_images"))
    ap.add_argument("--weights", type=Path, default=Path("cnn/runs/resnet18/best.pt"))
    ap.add_argument("--metrics", type=Path, default=Path("cnn/runs/resnet18/metrics.json"))
    ap.add_argument("--out", type=Path, default=Path("cnn_predictions.csv"))
    args = ap.parse_args()

    meta = json.loads(args.metrics.read_text())
    classes, size = meta["classes"], meta["input_size"]
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(torch.load(args.weights, map_location="cuda"))
    model.cuda().eval()
    tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                             transforms.Normalize(MEAN, STD)])

    files = sorted(p for p in args.images.rglob("*") if p.suffix.lower() in EXTS)
    rows, lat, correct = [], [], 0
    with torch.no_grad():
        model(torch.zeros(1, 3, size, size, device="cuda"))  # warm-up
        for p in files:
            t = time.perf_counter()
            x = tf(Image.open(p).convert("RGB")).unsqueeze(0).cuda()
            prob = model(x).softmax(1)[0]
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t) * 1000  # includes image decode + preprocessing
            k = int(prob.argmax())
            true = p.parent.name
            correct += classes[k] == true
            lat.append(ms)
            rows.append({"image": p.name, "predicted_label": classes[k], "confidence": f"{prob[k]:.4f}",
                         "true_label": true, "latency_ms": f"{ms:.1f}"})

    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{correct}/{len(rows)} correct = {correct / len(rows):.1%}; "
          f"latency mean {statistics.mean(lat):.1f} ms, median {statistics.median(lat):.1f} ms")


if __name__ == "__main__":
    main()
