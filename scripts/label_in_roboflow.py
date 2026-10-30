"""Give every training image in the Roboflow project its class label and add it to the Dataset.

Label source:
  - this laptop's batch (tagged at upload): the tag, using the image id recorded in upload_log.csv
    (stray second copies from the interrupted full-size upload are skipped);
  - the other uploads: the filename prefix, D = defective, ND = defect_free, W = wasted.
Wasted images are left out: the assignment has two classes.

    python scripts/label_in_roboflow.py list      # fetch ids + names -> scripts/roboflow_images.csv
    python scripts/label_in_roboflow.py plan      # decide labels -> scripts/label_plan.csv
    python scripts/label_in_roboflow.py apply [N] # annotate (first N only, for a test)
"""
import csv
import re
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "scripts"
WORKSPACE, PROJECT = "nirupama-sinha", "fabric-defect-identification"
TAG_LABEL = {"defective": "defective", "defect_free": "defect_free", "wasted": None}


def api_key():
    for line in (ROOT / ".env").read_text(encoding="utf-8-sig").splitlines():
        if line.startswith("ROBOFLOW_API_KEY="):
            return line.split("=", 1)[1].strip()
    sys.exit("ROBOFLOW_API_KEY missing from .env")


def list_images(key):
    rows, offset = [], 0
    while True:
        r = requests.post(
            f"https://api.roboflow.com/{WORKSPACE}/{PROJECT}/search",
            params={"api_key": key},
            json={"limit": 250, "offset": offset, "fields": ["id", "name", "tags"]},
            timeout=60,
        )
        r.raise_for_status()
        batch = r.json()["results"]
        rows += batch
        offset += len(batch)
        print(offset, flush=True)
        if len(batch) < 250:
            break
    with (HERE / "roboflow_images.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "name", "tags"])
        for x in rows:
            w.writerow([x["id"], x["name"], "|".join(x.get("tags") or [])])
    print(f"{len(rows)} images listed")


def prefix_label(name):
    name = name.upper()  # a few files use lower-case prefixes (dimage…, ndimage…)
    if name.startswith("ND"):
        return "defect_free"
    if name.startswith("D"):
        return "defective"
    if name.startswith("W"):
        return None
    return "unknown"


def plan():
    with (HERE / "upload_log.csv").open() as f:
        mine = {r["id"] for r in csv.DictReader(f) if r["status"] == "ok"}
    out, why = [], Counter()
    with (HERE / "roboflow_images.csv").open() as f:
        for r in csv.DictReader(f):
            tags = [t for t in r["tags"].split("|") if t in TAG_LABEL]
            if tags:
                if r["id"] not in mine:
                    why["skip: stray second copy"] += 1
                    continue
                label, source = TAG_LABEL[tags[0]], "laptop batch"
            else:
                label, source = prefix_label(r["name"]), "friends"
            if label is None:
                why[f"skip: wasted ({source})"] += 1
                continue
            if label == "unknown":
                why["skip: unknown prefix"] += 1
                print("unknown prefix:", r["name"])
                continue
            why[f"{label} ({source})"] += 1
            out.append([r["id"], r["name"], label, source])
    # Same capture frame uploaded by both sources -> keep one (the stem after the prefix).
    seen, dedup = set(), []
    for row in out:
        stem = re.sub(r"^(ND|D|W)", "", row[1], flags=re.I)
        if stem in seen:
            why["skip: same frame from both sources"] += 1
            continue
        seen.add(stem)
        dedup.append(row)
    with (HERE / "label_plan.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "name", "label", "source"])
        w.writerows(dedup)
    for k, v in sorted(why.items()):
        print(f"{v:6d}  {k}")
    print(f"{len(dedup)} images to label")


def apply(key, limit=None):
    with (HERE / "label_plan.csv").open() as f:
        rows = list(csv.DictReader(f))
    log_path = HERE / "label_log.csv"
    done = set()
    if log_path.exists():
        with log_path.open() as f:
            done = {r["id"] for r in csv.DictReader(f) if r["status"] == "ok"}
    rows = [r for r in rows if r["id"] not in done][:limit]
    print(f"{len(rows)} to label, {len(done)} already done", flush=True)
    new = not log_path.exists()
    logf = log_path.open("a", newline="")
    w = csv.writer(logf)
    if new:
        w.writerow(["id", "label", "status", "detail"])
    lock, n = threading.Lock(), [0]

    def one(r):
        status, detail = "error", ""
        for _ in range(3):
            try:
                resp = requests.post(
                    f"https://api.roboflow.com/dataset/{PROJECT}/annotate/{r['id']}",
                    params={"api_key": key, "name": Path(r["name"]).stem + ".txt", "overwrite": "true"},
                    json={"annotationFile": r["label"]},
                    timeout=60,
                )
                if resp.ok:
                    status, detail = "ok", ""
                    break
                detail = resp.text[:200]
            except Exception as e:
                detail = repr(e)[:200]
        with lock:
            w.writerow([r["id"], r["label"], status, detail])
            logf.flush()
            n[0] += 1
            if n[0] % 250 == 0 or status != "ok":
                print(n[0], r["name"], status, detail, flush=True)

    with ThreadPoolExecutor(max_workers=48) as pool:
        list(pool.map(one, rows))
    logf.close()
    print("FINISHED", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "list":
        list_images(api_key())
    elif cmd == "plan":
        plan()
    elif cmd == "apply":
        apply(api_key(), int(sys.argv[2]) if len(sys.argv) > 2 else None)
