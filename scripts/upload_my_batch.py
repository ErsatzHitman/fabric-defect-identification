"""Upload the 1,699-image batch to the Roboflow project, skipping exact duplicates.

Files are renamed with the same prefix convention the other uploads use
(D = Defected, ND = Defect Free, W = Wasted) and tagged with their label.
Resumable: names already uploaded are recorded in upload_log.csv.
"""
import csv
import io
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "fabric_handoff"
LOG = ROOT / "scripts" / "upload_log.csv"
PROJECT = "fabric-defect-identification"
BATCH = "Folder: Achintya batch"
PREFIX = {"Defected": "D", "Defect Free": "ND", "Wasted": "W"}
TAG = {"Defected": "defective", "Defect Free": "defect_free", "Wasted": "wasted"}
UPLOAD_WIDTH = 1024


def api_key():
    for line in (ROOT / ".env").read_text(encoding="utf-8-sig").splitlines():
        if line.startswith("ROBOFLOW_API_KEY="):
            return line.split("=", 1)[1].strip()
    sys.exit("ROBOFLOW_API_KEY missing from .env")


def shrink(path):
    """Full frames are ~7 MB; resize to UPLOAD_WIDTH so the upload fits a slow connection."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((UPLOAD_WIDTH, UPLOAD_WIDTH), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=92)
    return buf.getvalue()


def main():
    key = api_key()
    done = set()
    if LOG.exists():
        with LOG.open() as f:
            done = {r["name"] for r in csv.DictReader(f) if r["status"] == "ok"}

    jobs = []
    with (DATA / "labels.csv").open() as f:
        for r in csv.DictReader(f):
            if r["duplicate_of"]:
                continue
            name = PREFIX[r["label"]] + r["filename"]
            if name not in done:
                jobs.append((DATA / r["label"] / r["filename"], name, r["label"]))
    print(f"{len(jobs)} to upload, {len(done)} already done", flush=True)

    new_log = not LOG.exists()
    lock = threading.Lock()
    logf = LOG.open("a", newline="")
    writer = csv.writer(logf)
    if new_log:
        writer.writerow(["name", "label", "status", "id"])
    counter = [0]

    def upload(job):
        path, name, label = job
        status, rid = "error", "not attempted"
        img = shrink(path)
        for _ in range(3):
            try:
                resp = requests.post(
                    f"https://api.roboflow.com/dataset/{PROJECT}/upload",
                    params={"api_key": key, "name": name, "batch": BATCH, "tag": TAG[label]},
                    files={"file": (name, img, "image/jpeg")},
                    timeout=120,
                )
                body = resp.json()
                if resp.ok and (body.get("success") or body.get("duplicate")):
                    status, rid = "ok", body.get("id", "")
                    break
                status, rid = "error", str(body)[:200]
            except Exception as e:  # network hiccup: retry
                status, rid = "error", repr(e)[:200]
        with lock:
            writer.writerow([name, label, status, rid])
            logf.flush()
            counter[0] += 1
            if counter[0] % 50 == 0 or status != "ok":
                print(f"{counter[0]}/{len(jobs)} {name} {status} {rid if status != 'ok' else ''}", flush=True)

    with ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(upload, jobs))
    logf.close()
    print("FINISHED", flush=True)


if __name__ == "__main__":
    main()
