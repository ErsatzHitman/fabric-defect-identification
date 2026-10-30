"""Fine-tune ResNet18 (ImageNet weights) on the same train/valid/test split exported from Roboflow.

Expected layout (Roboflow "folder" classification export):
    DATA/train/<class>/*.jpg, DATA/valid/<class>/*.jpg, DATA/test/<class>/*.jpg

Images are kept at 512 px: at 224 px the tiny white defects disappear.
Writes metrics.json, test_predictions.csv and best.pt to OUT.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def loaders(data, size, batch):
    train_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    sets = {
        "train": datasets.ImageFolder(data / "train", train_tf),
        "valid": datasets.ImageFolder(data / "valid", eval_tf),
        "test": datasets.ImageFolder(data / "test", eval_tf),
    }
    out = {
        k: DataLoader(v, batch_size=batch, shuffle=(k == "train"), num_workers=4,
                      pin_memory=True, persistent_workers=True)
        for k, v in sets.items()
    }
    return sets, out


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    preds, confs = [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            p = model(x).float().softmax(1)
        c, k = p.max(1)
        correct += (k == y).sum().item()
        total += y.numel()
        preds += k.tolist()
        confs += c.tolist()
    return correct / total, preds, confs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("cnn/runs/resnet18"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    device = "cuda"
    sets, dl = loaders(args.data, args.size, args.batch)
    classes = sets["train"].classes
    print("classes", classes, {k: len(v) for k, v in sets.items()}, flush=True)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.to(device)

    # Weight the loss by inverse class frequency: the data is roughly 70 % defective.
    counts = torch.bincount(torch.tensor(sets["train"].targets), minlength=len(classes)).float()
    loss_fn = nn.CrossEntropyLoss(weight=(counts.sum() / counts).to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * len(dl["train"]))
    scaler = torch.amp.GradScaler()

    best, history = 0.0, []
    t0 = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for x, y in dl["train"]:
            x, y = x.to(device, non_blocking=True), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item() * y.numel()
        val_acc, _, _ = evaluate(model, dl["valid"], device)
        history.append({"epoch": epoch + 1, "loss": running / len(sets["train"]), "val_acc": val_acc})
        print(history[-1], f"{time.perf_counter() - t0:.0f}s", flush=True)
        if val_acc >= best:
            best = val_acc
            torch.save(model.state_dict(), args.out / "best.pt")
    train_seconds = time.perf_counter() - t0

    model.load_state_dict(torch.load(args.out / "best.pt"))
    test_acc, preds, confs = evaluate(model, dl["test"], device)

    # Per-image latency: batch size 1, after warm-up, including the GPU sync.
    model.eval()
    x = sets["test"][0][0].unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(10):
            model(x)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(100):
            model(x)
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - t) / 100 * 1000

    n = len(classes)
    cm = [[0] * n for _ in range(n)]
    with (args.out / "test_predictions.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "true_label", "predicted_label", "confidence"])
        for (path, y), k, c in zip(sets["test"].samples, preds, confs):
            cm[y][k] += 1
            w.writerow([Path(path).name, classes[y], classes[k], f"{c:.4f}"])

    metrics = {
        "model": "resnet18 (ImageNet fine-tune)",
        "input_size": args.size,
        "epochs": args.epochs,
        "classes": classes,
        "split_sizes": {k: len(v) for k, v in sets.items()},
        "best_val_acc": best,
        "test_acc": test_acc,
        "confusion_matrix_rows_true_cols_pred": cm,
        "train_seconds": train_seconds,
        "latency_ms_per_image_gpu": latency_ms,
        "gpu": torch.cuda.get_device_name(0),
        "history": history,
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
