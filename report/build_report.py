"""Build report/report.pdf (3 pages) from report/summary.json and the figures."""
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

REP = Path(__file__).resolve().parent
S = json.loads((REP / "summary.json").read_text())
NOTES = json.loads((REP / "notes.json").read_text())  # analysis paragraphs written after reading results

ss = getSampleStyleSheet()
H1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=15, spaceAfter=4)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11.5, spaceBefore=6, spaceAfter=3)
P = ParagraphStyle("p", parent=ss["BodyText"], fontSize=9.2, leading=12.2, spaceAfter=4)
CELL = ParagraphStyle("cell", parent=ss["BodyText"], fontSize=8.5, leading=10.5)
SMALL = ParagraphStyle("s", parent=P, fontSize=8, leading=10, textColor=colors.HexColor("#444444"))


def pct(x):
    return f"{100 * x:.1f}%"


def table(rows, widths, header=True):
    # Wrap every cell in a Paragraph so long text breaks inside its column and entities render.
    rows = [[Paragraph(f"<b>{c}</b>" if header and i == 0 else str(c), CELL) for c in row]
            for i, row in enumerate(rows)]
    t = Table(rows, colWidths=widths)
    style = [("FONTSIZE", (0, 0), (-1, -1), 8.5), ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 2),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef7")),
                  ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    t.setStyle(TableStyle(style))
    return t


def fig(name, width):
    """Place a figure at the given width, keeping its aspect ratio."""
    from PIL import Image as PILImage
    w, h = PILImage.open(REP / name).size
    return Image(str(REP / name), width=width, height=width * h / w)


rf, cn = S["roboflow"], S["cnn"]
rft = rf["training"]
story = [
    Paragraph("Fabric Defect Detection: Roboflow vs. a Custom PyTorch CNN", H1),
    Paragraph("Assignment 1 &middot; code: github.com/ErsatzHitman/fabric-defect-identification", SMALL),
    Spacer(1, 4),
    Paragraph("1. Task and data", H2),
    Paragraph(NOTES["data"], P),
    table([
        ["", "defective", "defect_free", "total"],
        ["Train (70%)", "2,440", "991", "3,431"],
        ["Valid (20%)", "698", "282", "980"],
        ["Test (10%)", "352", "138", "490"],
        ["Unseen sample for infer.py", "36", "14", "50"],
    ], [5 * cm, 2.6 * cm, 2.6 * cm, 2.2 * cm]),
    Spacer(1, 3),
    Paragraph(NOTES["split"], P),
    Paragraph("2. The two pipelines", H2),
    table([
        ["", "Roboflow (no-code)", "Custom CNN (PyTorch)"],
        ["Model", "ViT-Base/16, ImageNet-21k pretrained", "ResNet18, ImageNet-1k pretrained"],
        ["Input", "512x512 version, resized to 224 by the model", "512x512 (tiny defects vanish at 224)"],
        ["Augmentation", "Flip H+V, brightness &plusmn;15% (2 copies / image)",
         "Same exported copies + online flip, brightness/contrast jitter"],
        ["Training", f"Roboflow hosted GPU, {rft['epochs']} epochs",
         f"RTX 5060 laptop GPU, {cn['epochs']} epochs, AdamW, one-cycle LR, class-weighted loss"],
        ["Inference", "Hosted REST API (infer.py)", "Local GPU (cnn/predict_cnn.py)"],
    ], [2.6 * cm, 6.6 * cm, 7.4 * cm]),
    Paragraph(NOTES["pipelines"], P),
    PageBreak(),
    Paragraph("3. Results", H2),
    table([
        ["Metric", "Roboflow ViT-B/16", "ResNet18 (PyTorch)"],
        ["Accuracy, full test split (n=490)", pct(rf["test490"]["accuracy"]), pct(cn["test490"]["accuracy"])],
        ["Accuracy, 50 unseen images (infer.py)", pct(rf["test50"]["accuracy"]), pct(cn["test50"]["accuracy"])],
        ["Recall, defective", pct(rf["test490"]["recall_defective"]), pct(cn["test490"]["recall_defective"])],
        ["Recall, defect_free", pct(rf["test490"]["recall_defect_free"]), pct(cn["test490"]["recall_defect_free"])],
        ["Precision, defective", pct(rf["test490"]["precision_defective"]), pct(cn["test490"]["precision_defective"])],
        ["F1, defective", f"{rf['test490']['f1_defective']:.3f}", f"{cn['test490']['f1_defective']:.3f}"],
        ["Training time (wall clock)", f"{rft['minutes']:.0f} min (incl. queue)", f"{cn['train_seconds'] / 60:.1f} min"],
        ["Latency per image, end to end (median, n=50)", f"{rf['latency50']['median_ms']:.0f} ms (network API)",
         f"{cn['latency50']['median_ms']:.0f} ms (decode + GPU)"],
        ["Latency, model forward pass only", "n/a (hosted)", f"{cn['gpu_forward_ms']:.1f} ms"],
    ], [7.2 * cm, 4.6 * cm, 4.8 * cm]),
    Spacer(1, 4),
    fig("fig_confusion.png", 14 * cm),
    Paragraph("Figure 1. Confusion matrices on the 490-image test split.", SMALL),
    Paragraph(NOTES["results"], P),
    Paragraph(NOTES["tradeoffs"], P),
    PageBreak(),
    Paragraph("4. Failure analysis of misclassified edge cases", H2),
    fig("fig_errors.png", 15.5 * cm),
    Paragraph("Figure 2. Test images misclassified by at least one model (RF = Roboflow; confidence in brackets).",
              SMALL),
    Paragraph(NOTES["failures"], P),
    Paragraph("5. Conclusion", H2),
    Paragraph(NOTES["conclusion"], P),
]

SimpleDocTemplate(str(REP / "report.pdf"), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                  topMargin=1.6 * cm, bottomMargin=1.6 * cm,
                  title="Fabric Defect Detection: Roboflow vs. Custom CNN").build(story)
print("wrote", REP / "report.pdf")
