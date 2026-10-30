# Fabric Defect Detection: Roboflow vs. a custom PyTorch CNN

Assignment 1: an end-to-end computer-vision pipeline that classifies fabric images as
`defective` or `defect_free`, built twice, once on Roboflow (no-code) and once as a fine-tuned
ResNet18 in PyTorch, and benchmarked on the same split.

## Data

| Source | Images | Labelled by |
|---|---|---|
| Batch 1 | 1,699 | rule-based detector + manual review (see below) |
| Batch 2 | 3,326 | team members, by hand |
| **Unique images** | **4,999** | |

Images are 2448×2048 RGB frames of moving fabric under a fixed camera. Three categories were labelled
(Defected, Defect Free, Wasted). The assignment has two classes, so **Wasted** frames (creases, the
instructor's hand, fabric joints; 98 images) are excluded, leaving **4,901 images**:

| Class | Images |
|---|---|
| `defective` | 3,490 (71 %) |
| `defect_free` | 1,411 (29 %) |

Labelling rules: any white mark that stands out from the fabric is a defect (even if the fabric is
also creased); a crease with no defect is Wasted; dark marks are lens dust, not defects.
26 byte-identical duplicate files were removed before upload.

**Split (fixed once, in Roboflow, shared by both models):** 70/20/10 → 3,431 train / 980 valid / 490 test.
`infer.py` is run on 50 images drawn from the test split, which neither model ever trained on.

## Repository layout

| Path | Purpose |
|---|---|
| `scripts/upload_my_batch.py` | Upload batch 1 to Roboflow (resized to 1024 px for a slow uplink), skipping duplicates |
| `scripts/label_in_roboflow.py` | Apply class labels to every image and add it to the Roboflow Dataset |
| `scripts/fetch_split.py` | Download the generated Roboflow version (same split) and pick the 50 test images |
| `infer.py` | Roboflow API inference on the 50 test images → `predictions.csv` |
| `cnn/train_cnn.py` | Fine-tune ResNet18 on the same split; writes metrics and test predictions |
| `report/` | 3-page PDF report and the script that builds it |

## Reproduce

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # torch: use the CUDA wheel for your GPU
echo ROBOFLOW_API_KEY=... > .env                                         # never commit this file

python scripts/fetch_split.py --version 1          # data/v1/{train,valid,test} + test_images/
python infer.py --images test_images --version 1   # predictions.csv
python cnn/train_cnn.py --data data/v1             # cnn/runs/resnet18/metrics.json
```

## Roboflow setup

- Project: `fabric-defect-identification` (single-label classification)
- Preprocessing: auto-orient, resize to 512×512
- Augmentations (train only, 2 outputs per image): horizontal + vertical flip, brightness ±15 %
- Model: ViT-Base/16 (ImageNet-21k pretrained), Roboflow hosted training

See `report/report.pdf` for results, the comparison and the failure analysis.
