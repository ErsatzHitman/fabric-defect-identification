"""Download the Roboflow dataset version (folder format) so the CNN trains on exactly the same split,
then copy 50 test images into test_images/<class>/ for infer.py.

    python scripts/fetch_split.py --version 1
"""
import argparse
import io
import random
import shutil
import sys
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE, PROJECT = "nirupama-sinha", "fabric-defect-identification"


def api_key():
    for line in (ROOT / ".env").read_text(encoding="utf-8-sig").splitlines():
        if line.startswith("ROBOFLOW_API_KEY="):
            return line.split("=", 1)[1].strip()
    sys.exit("ROBOFLOW_API_KEY missing from .env")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", type=int, required=True)
    ap.add_argument("--n-test", type=int, default=50)
    args = ap.parse_args()

    key = api_key()
    meta = requests.get(f"https://api.roboflow.com/{WORKSPACE}/{PROJECT}/{args.version}/folder",
                        params={"api_key": key}, timeout=120).json()
    link = meta["export"]["link"]
    print("downloading export ...", flush=True)
    blob = requests.get(link, timeout=1800).content
    out = ROOT / "data" / f"v{args.version}"
    if out.exists():
        shutil.rmtree(out)
    zipfile.ZipFile(io.BytesIO(blob)).extractall(out)
    for split in ("train", "valid", "test"):
        counts = {d.name: len(list(d.glob("*"))) for d in sorted((out / split).iterdir()) if d.is_dir()}
        print(split, counts)

    # 50 unseen images from the test split, stratified by class, fixed seed.
    test = out / "test"
    classes = sorted(d.name for d in test.iterdir() if d.is_dir())
    total = sum(len(list((test / c).glob("*"))) for c in classes)
    dest = ROOT / "test_images"
    if dest.exists():
        shutil.rmtree(dest)
    rng = random.Random(42)
    picked = 0
    for i, c in enumerate(classes):
        files = sorted((test / c).glob("*"))
        k = args.n_test - picked if i == len(classes) - 1 else round(args.n_test * len(files) / total)
        (dest / c).mkdir(parents=True)
        for f in rng.sample(files, k):
            shutil.copy2(f, dest / c / f.name)
        picked += k
        print(f"test_images/{c}: {k}")


if __name__ == "__main__":
    main()
