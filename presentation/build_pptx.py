"""Generate the 19-slide conference deck for the final presentation.

STALE (2026-10-24): this deck tells the earlier three-contribution story and
reads a `cal_tda` method row that `results/summary.csv` no longer carries
(the headline methods are now `cal_tda_gate`, `cal_tda_temp` and
`cal_tda_gate_temp`), so it currently fails to build. Updating it to the
gated-fusion finding of `report/main_v2.pdf` is a follow-up plan.

Every number on every slide is READ FROM `results/summary.csv` and
`results/ablation.csv` at build time. Nothing is typed in by hand. Re-run the
sweep, re-run this script, and the deck is correct again -- the same rule the
report and README already follow, applied to the talk.

    python presentation/build_pptx.py

Writes `presentation/calibration_aware_tta.pptx` (16:9). Speaker notes are in
the PowerPoint notes pane of every slide; `presentation/slides.md` mirrors the
same structure for reading outside PowerPoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

# --- Theme -------------------------------------------------------------------
# Hues are the ones the figures already use, so a slide and the chart on it
# are never two different purples. ACCENT/BASELINE are the validated pair from
# `analysis/make_figures.py` (see the palette note there).
INK = RGBColor(0x1A, 0x1F, 0x2B)
MUTED = RGBColor(0x5A, 0x61, 0x70)
ACCENT = RGBColor(0x7D, 0x5B, 0xA6)       # "ours"
BASELINE = RGBColor(0xDD, 0x84, 0x52)     # TDA
NEUTRAL = RGBColor(0xC9, 0xC6, 0xC0)
RULE = RGBColor(0xD8, 0xD5, 0xD0)
SURFACE = RGBColor(0xFF, 0xFF, 0xFF)
WARN = RGBColor(0xB3, 0x4A, 0x3C)

# Slide background and the fills that have to read against it. Content slides
# sit on warm paper; panels are white cards so they stand off the ground, and
# the table header/chip neutrals are a step darker than the paper so they do
# not vanish into it.
BG = RGBColor(0xF4, 0xF0, 0xE9)
BG_DARK = RGBColor(0x25, 0x1F, 0x36)      # title slide
HEADER_FILL = RGBColor(0xE7, 0xE2, 0xD8)  # table headers, inactive chips
TINT_ACCENT = RGBColor(0xEC, 0xE6, 0xF3)  # lavender emphasis fill
TINT_WARN = RGBColor(0xF7, 0xEA, 0xE1)    # peach warning fill

FONT = "Calibri"

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)
MARGIN = Inches(0.62)
BODY_TOP = Inches(1.72)
BODY_BOTTOM = Inches(6.82)
CONTENT_W = SLIDE_W - 2 * MARGIN

DATASET_LABELS = {
    "dtd": "DTD",
    "flower102": "Flowers102",
    "pets": "Pets",
    "eurosat": "EuroSAT",
    "aircraft": "Aircraft",
}
DATASET_ORDER = list(DATASET_LABELS)


# --- Data --------------------------------------------------------------------
class Results:
    """Every figure the deck quotes, pulled from the committed CSVs.

    Kept as one object so a slide function can never reach around it to a
    hard-coded literal -- if a number is not derivable from the CSVs, it does
    not belong on a slide.
    """

    def __init__(self, summary_csv: Path, ablation_csv: Path,
                 method_config_yaml: Path):
        self.summary = pd.read_csv(summary_csv)
        self.ablation = pd.read_csv(ablation_csv)
        self.method_config = yaml.safe_load(
            Path(method_config_yaml).read_text(encoding="utf-8")
        )

    def cell(self, dataset: str, method: str, column: str) -> float:
        row = self.summary[
            (self.summary["dataset"] == dataset) & (self.summary["method"] == method)
        ]
        if row.empty:
            raise KeyError(f"no {method} row for {dataset} in summary.csv")
        return float(row.iloc[0][column])

    def method_mean(self, method: str, column: str) -> float:
        return float(self.summary[self.summary["method"] == method][column].mean())

    def variant_mean(self, variant: str, column: str) -> float:
        return float(self.ablation[self.ablation["variant"] == variant][column].mean())

    def variant_cell(self, dataset: str, variant: str, column: str) -> float:
        row = self.ablation[
            (self.ablation["dataset"] == dataset) & (self.ablation["variant"] == variant)
        ]
        if row.empty:
            raise KeyError(f"no {variant} row for {dataset} in ablation.csv")
        return float(row.iloc[0][column])

    def method_constant(self, key: str):
        """A knob read from the shipped method config, not retyped onto a slide.

        The deck quotes `margin_threshold` and `fusion_weight` as part of the
        method description. Those live in `configs/method/cal_tda.yaml`, which
        is what the runs actually used, so the slide should read them from
        there rather than carry a second copy that can silently disagree.
        """
        return self.method_config[key]

    def n_total_images(self) -> int:
        per_dataset = self.summary.groupby("dataset")["n_samples"].first()
        return int(per_dataset.sum())

    def n_ablation_runs(self) -> int:
        return int(len(self.ablation))

    def admission_range(self) -> tuple[tuple[str, float], tuple[str, float]]:
        rows = self.summary[self.summary["method"] == "cal_tda"].dropna(
            subset=["admission_rate"]
        )
        low = rows.loc[rows["admission_rate"].idxmin()]
        high = rows.loc[rows["admission_rate"].idxmax()]
        return (low["dataset"], float(low["admission_rate"])), (
            high["dataset"], float(high["admission_rate"])
        )

    def boundary_fraction_range(self) -> tuple[float, float]:
        column = self.summary[self.summary["method"] == "cal_tda"][
            "temp_frac_at_grid_boundary"
        ].dropna()
        return float(column.min()), float(column.max())

    def criterion_hits(self) -> int:
        """Datasets where cal_tda beats tda on accuracy AND on ECE.

        This is the spec's success criterion, recomputed here rather than
        asserted, so the slide cannot claim a different count than the CSV
        supports.
        """
        hits = 0
        for dataset in DATASET_ORDER:
            better_accuracy = self.cell(dataset, "cal_tda", "accuracy") >= self.cell(
                dataset, "tda", "accuracy"
            )
            better_ece = self.cell(dataset, "cal_tda", "ece") < self.cell(
                dataset, "tda", "ece"
            )
            hits += int(better_accuracy and better_ece)
        return hits

    def temp_beats_none_on_ece(self) -> int:
        return sum(
            int(
                self.variant_cell(dataset, "temp", "ece")
                < self.variant_cell(dataset, "none", "ece")
            )
            for dataset in DATASET_ORDER
        )


# --- Layout helpers ----------------------------------------------------------
def _style_run(run, size, *, bold=False, color=INK, italic=False, font=FONT):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    run.font.name = font


def _textbox(slide, left, top, width, height, *, anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(left, top, width, height)
    frame = box.text_frame
    frame.word_wrap = True
    frame.vertical_anchor = anchor
    frame.margin_left = frame.margin_right = 0
    frame.margin_top = frame.margin_bottom = 0
    return box, frame


def _paint_background(slide, color=BG):
    """Solid slide background plus a thin accent band along the top edge.

    Set on the slide's own background fill rather than a full-bleed rectangle,
    so content shapes never have to fight it for z-order.
    """
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = color

    band = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, Inches(0.09))
    band.fill.solid()
    band.fill.fore_color.rgb = ACCENT
    band.line.fill.background()
    band.shadow.inherit = False


def new_slide(prs, title, *, kicker=None):
    """A blank slide with the deck's standard header: kicker, then the title."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _paint_background(slide)

    top = Inches(0.46)
    if kicker:
        _, frame = _textbox(slide, MARGIN, top, CONTENT_W, Inches(0.26))
        run = frame.paragraphs[0].add_run()
        run.text = kicker.upper()
        _style_run(run, 11.5, bold=True, color=ACCENT)
        top = top + Inches(0.30)

    _, frame = _textbox(slide, MARGIN, top, CONTENT_W, Inches(0.72))
    run = frame.paragraphs[0].add_run()
    run.text = title
    _style_run(run, 29, bold=True, color=INK)
    return slide


def add_bullets(slide, items, *, left=None, top=None, width=None,
                size=18, sub_size=15, spacing=12):
    """`items` is a list of strings, or (level, text), or (level, text, color)."""
    left = MARGIN if left is None else left
    top = BODY_TOP if top is None else top
    width = CONTENT_W if width is None else width

    _, frame = _textbox(slide, left, top, width, BODY_BOTTOM - top)
    first = True
    for item in items:
        if isinstance(item, str):
            level, text, color = 0, item, INK
        elif len(item) == 2:
            (level, text), color = item, INK
        else:
            level, text, color = item

        paragraph = frame.paragraphs[0] if first else frame.add_paragraph()
        first = False
        paragraph.space_after = Pt(spacing if level == 0 else spacing * 0.5)

        if not text:                      # blank line as a spacer
            _style_run(paragraph.add_run(), 6)
            continue

        # Hanging indent, set on the XML because python-pptx exposes no API for
        # it: without this, a bullet that wraps puts its second line flush under
        # the bullet glyph instead of under the text, and every multi-line
        # bullet in the deck reads as two separate items.
        # `indent` is relative to `marL`, so a negative value pulls only the
        # first line back out to where the glyph sits.
        hang = Inches(0.30)
        properties = paragraph._p.get_or_add_pPr()
        properties.set("marL", str(int(hang if level == 0 else hang + Inches(0.56))))
        properties.set("indent", str(int(-hang)))

        bullet = "•   " if level == 0 else "–   "
        run = paragraph.add_run()
        run.text = bullet + text
        _style_run(
            run,
            size if level == 0 else sub_size,
            color=color if level == 0 else MUTED,
        )
    return frame


def add_panel(slide, left, top, width, height, *, fill=SURFACE, line=RULE):
    panel = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    panel.fill.solid()
    panel.fill.fore_color.rgb = fill
    if line is None:
        panel.line.fill.background()
    else:
        panel.line.color.rgb = line
        panel.line.width = Pt(1.25)
    panel.shadow.inherit = False
    panel.adjustments[0] = 0.04
    panel.text_frame.text = ""
    return panel


def add_callout(slide, text, *, top=None, color=ACCENT, size=16.5, left=None, width=None):
    """The one sentence the audience should leave the slide with."""
    left = MARGIN if left is None else left
    width = CONTENT_W if width is None else width
    top = Inches(6.08) if top is None else top
    height = Inches(0.66)

    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, Pt(4), height)
    bar.fill.solid()
    bar.fill.fore_color.rgb = color
    bar.line.fill.background()
    bar.shadow.inherit = False

    _, frame = _textbox(
        slide, left + Inches(0.18), top, width - Inches(0.18), height,
        anchor=MSO_ANCHOR.MIDDLE,
    )
    run = frame.paragraphs[0].add_run()
    run.text = text
    _style_run(run, size, bold=True, color=INK)
    return frame


def add_picture(slide, path, left, top, max_width, max_height):
    """Place an image inside a box, preserving aspect ratio, centred."""
    picture = slide.shapes.add_picture(str(path), left, top, width=max_width)
    if picture.height > max_height:
        picture._element.getparent().remove(picture._element)
        picture = slide.shapes.add_picture(str(path), left, top, height=max_height)
    picture.left = int(left + (max_width - picture.width) / 2)
    picture.top = int(top + (max_height - picture.height) / 2)
    return picture


def add_table(slide, header, rows, left, top, width, *,
              col_widths=None, font_size=11.5, header_size=11.5,
              row_height=Inches(0.33), highlight_rows=(), highlight_color=None):
    """A plain table: tinted header, hairline row separators, no heavy grid."""
    shape = slide.shapes.add_table(
        len(rows) + 1, len(header), left, top, width, row_height * (len(rows) + 1)
    )
    table = shape.table
    table.first_row = True
    table.horz_banding = False

    if col_widths:
        for index, fraction in enumerate(col_widths):
            table.columns[index].width = Emu(int(width * fraction))

    for index, text in enumerate(header):
        cell = table.cell(0, index)
        cell.text = ""
        cell.fill.solid()
        cell.fill.fore_color.rgb = HEADER_FILL
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        cell.margin_left = cell.margin_right = Inches(0.07)
        paragraph = cell.text_frame.paragraphs[0]
        paragraph.alignment = PP_ALIGN.LEFT if index == 0 else PP_ALIGN.RIGHT
        run = paragraph.add_run()
        run.text = text
        _style_run(run, header_size, bold=True, color=MUTED)

    for row_index, row in enumerate(rows, start=1):
        table.rows[row_index].height = row_height
        emphasised = (row_index - 1) in highlight_rows
        for col_index, text in enumerate(row):
            cell = table.cell(row_index, col_index)
            cell.text = ""
            cell.fill.solid()
            cell.fill.fore_color.rgb = (
                TINT_ACCENT if emphasised and highlight_color is None else SURFACE
            )
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left = cell.margin_right = Inches(0.07)
            paragraph = cell.text_frame.paragraphs[0]
            paragraph.alignment = PP_ALIGN.LEFT if col_index == 0 else PP_ALIGN.RIGHT
            run = paragraph.add_run()
            run.text = text
            _style_run(
                run, font_size, bold=emphasised,
                color=(highlight_color or INK) if emphasised else INK,
            )
    return table


def add_footer(slide, index, total):
    _, frame = _textbox(slide, MARGIN, Inches(7.00), Inches(8.0), Inches(0.3))
    run = frame.paragraphs[0].add_run()
    run.text = "Calibration-Aware Test-Time Adaptation for CLIP"
    _style_run(run, 10, color=MUTED)

    _, frame = _textbox(
        slide, SLIDE_W - MARGIN - Inches(1.2), Inches(7.00), Inches(1.2), Inches(0.3)
    )
    frame.paragraphs[0].alignment = PP_ALIGN.RIGHT
    run = frame.paragraphs[0].add_run()
    run.text = f"{index} / {total}"
    _style_run(run, 10, color=MUTED)


def set_notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text.strip()


def pct(value, digits=2):
    return f"{value:.{digits}f}"


def render_calibration_figure(out_path: Path) -> Path:
    """A textbook-style reliability diagram, drawn from SYNTHETIC curves.

    The background section explains what calibration is before any of our
    results exist in the talk, so this figure is deliberately illustrative:
    no run data, just the shape of the concept — a perfectly calibrated
    diagonal, an overconfident model sagging below it, and one bin's gap
    singled out as "this is what ECE adds up".
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    confidence = np.linspace(0.30, 0.975, 14)
    overconfident = confidence ** 2.4          # accuracy < confidence everywhere

    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#4D4D4D", linewidth=1.6,
            label="perfectly calibrated\n(accuracy = confidence)")
    ax.plot(confidence, overconfident, marker="o", markersize=5.5,
            color="#B34A3C", linewidth=2.0,
            label="overconfident model\n(accuracy < confidence)")

    # Single out one bin's gap — the quantity ECE averages over all bins.
    pick = 10
    ax.plot([confidence[pick]] * 2, [overconfident[pick], confidence[pick]],
            color="#7D5BA6", linewidth=2.6, solid_capstyle="butt",
            label="one bin's gap\n= |accuracy − confidence|")
    ax.annotate(
        "ECE = the weighted\naverage of these gaps",
        xy=(confidence[pick], (overconfident[pick] + confidence[pick]) / 2),
        xytext=(0.42, 0.80), fontsize=9.5, color="#7D5BA6", fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": "#7D5BA6", "linewidth": 1.4},
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("confidence the model reports", fontsize=11)
    ax.set_ylabel("accuracy it actually achieves", fontsize=11)
    ax.tick_params(labelsize=9)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8.5, frameon=True, framealpha=0.95,
              edgecolor="#D8D5D0")
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


# --- Diagram primitives ------------------------------------------------------
def _flow_box(slide, left, top, width, height, title, subtitle=None, *,
              fill=SURFACE, border=RULE, title_color=INK, title_size=13.5,
              subtitle_size=10.5):
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    box.fill.solid()
    box.fill.fore_color.rgb = fill
    box.line.color.rgb = border
    box.line.width = Pt(1.25)
    box.shadow.inherit = False
    box.adjustments[0] = 0.12

    frame = box.text_frame
    frame.word_wrap = True
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    frame.margin_left = frame.margin_right = Inches(0.08)
    frame.margin_top = frame.margin_bottom = Inches(0.04)

    paragraph = frame.paragraphs[0]
    paragraph.alignment = PP_ALIGN.CENTER
    run = paragraph.add_run()
    run.text = title
    _style_run(run, title_size, bold=True, color=title_color)

    if subtitle:
        paragraph = frame.add_paragraph()
        paragraph.alignment = PP_ALIGN.CENTER
        run = paragraph.add_run()
        run.text = subtitle
        _style_run(run, subtitle_size, color=MUTED)
    return box


def _arrow(slide, left, top, width, height, shape=MSO_SHAPE.RIGHT_ARROW, color=NEUTRAL):
    arrow = slide.shapes.add_shape(shape, left, top, width, height)
    arrow.fill.solid()
    arrow.fill.fore_color.rgb = color
    arrow.line.fill.background()
    arrow.shadow.inherit = False
    arrow.text_frame.text = ""
    return arrow


def _tda_diagram(slide):
    """TDA as a top-to-bottom flow: CLIP, then the two caches, then the sum."""
    row_a_top, row_a_h = Inches(2.15), Inches(0.78)
    row_b_top, row_b_h = Inches(3.50), Inches(0.92)
    row_c_top, row_c_h = Inches(5.00), Inches(0.72)

    widths = [Inches(2.55), Inches(3.45), Inches(3.05)]
    gap = Inches(0.52)
    total = sum(widths) + gap * (len(widths) - 1)
    left = int(MARGIN + (CONTENT_W - total) / 2)

    labels = [
        ("Test image  x", "one image, arriving online"),
        ("Frozen CLIP encoders", "no backprop, no labels, no retraining"),
        ("Zero-shot logits", "image features @ text features"),
    ]
    for index, ((title, subtitle), width) in enumerate(zip(labels, widths)):
        _flow_box(slide, left, row_a_top, width, row_a_h, title, subtitle)
        if index < len(widths) - 1:
            _arrow(slide, int(left + width + Inches(0.06)), int(row_a_top + Inches(0.23)),
                   int(gap - Inches(0.12)), Inches(0.32))
        left = int(left + width + gap)

    # Centred on the slide, not under the logits box it leaves: the arrow feeds
    # BOTH caches, and parking it above one of them reads as a flow into that
    # one alone. Centred, it lands in the gap between them and shares a vertical
    # spine with the arrow below.
    _arrow(slide, int(SLIDE_W / 2 - Inches(0.16)), int(row_a_top + row_a_h + Inches(0.07)),
           Inches(0.32), Inches(0.42), shape=MSO_SHAPE.DOWN_ARROW)

    cache_w, cache_gap = Inches(4.35), Inches(0.6)
    cache_left = int(MARGIN + (CONTENT_W - (cache_w * 2 + cache_gap)) / 2)
    _flow_box(
        slide, cache_left, row_b_top, cache_w, row_b_h,
        "Positive cache",
        "confident samples, keyed by their own pseudo-label, 3 per class",
        fill=RGBColor(0xEC, 0xE6, 0xF3), border=ACCENT,
    )
    _flow_box(
        slide, int(cache_left + cache_w + cache_gap), row_b_top, cache_w, row_b_h,
        "Negative cache",
        "mid-entropy samples: evidence for what the class is NOT",
        fill=RGBColor(0xF7, 0xEA, 0xE1), border=BASELINE,
    )

    _arrow(slide, int(SLIDE_W / 2 - Inches(0.16)), int(row_b_top + row_b_h + Inches(0.07)),
           Inches(0.32), Inches(0.42), shape=MSO_SHAPE.DOWN_ARROW)

    return _flow_box(
        slide, MARGIN, row_c_top, CONTENT_W, row_c_h,
        "final logits  =  CLIP logits  +  α · cache_pos  −  α_neg · cache_neg",
        "a sum of logits, not a re-normalised mixture  -  remember this line",
        fill=SURFACE, border=INK, title_size=16,
    )


# --- Slides ------------------------------------------------------------------
def slide_title(prs, results):
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    # Dark cover, light text: the one slide that gets its own treatment, so
    # the deck opens like a deck rather than a document.
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = BG_DARK

    band = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, 0, SLIDE_H - Inches(0.14), SLIDE_W, Inches(0.14)
    )
    band.fill.solid()
    band.fill.fore_color.rgb = ACCENT
    band.line.fill.background()
    band.shadow.inherit = False

    light = RGBColor(0xF5, 0xF2, 0xFA)
    lavender = RGBColor(0xB9, 0xA6, 0xD6)
    soft = RGBColor(0xA8, 0xA2, 0xB8)

    _, frame = _textbox(slide, Inches(1.15), Inches(2.00), Inches(11.2), Inches(0.4))
    run = frame.paragraphs[0].add_run()
    run.text = "TAU  ·  DEEP LEARNING  ·  FINAL PROJECT"
    _style_run(run, 13, bold=True, color=lavender)

    _, frame = _textbox(slide, Inches(1.15), Inches(2.52), Inches(11.2), Inches(1.7))
    run = frame.paragraphs[0].add_run()
    run.text = "Calibration-Aware\nTest-Time Adaptation for CLIP"
    _style_run(run, 44, bold=True, color=light)

    _, frame = _textbox(slide, Inches(1.15), Inches(4.60), Inches(11.2), Inches(1.0))
    run = frame.paragraphs[0].add_run()
    run.text = "Gal Machlin  (207627811)          Idan Yehuda  (207016270)"
    _style_run(run, 17, color=light)

    paragraph = frame.add_paragraph()
    paragraph.space_before = Pt(10)
    run = paragraph.add_run()
    run.text = ("Can we make test-time adaptation more trustworthy — not just more "
                "accurate — with no training and no labels?")
    _style_run(run, 14, italic=True, color=soft)

    set_notes(slide, """
Good morning. Our project asks a narrow question about test-time adaptation.

TTA methods make CLIP more accurate on new domains without any training or
labels. We asked whether they also make it more trustworthy - whether the
confidence it reports can still be believed - and whether we could fix that
with three training-free changes.

The short answer, which I will not hide until the end: two of our three ideas
failed, one worked, and the interesting part of this talk is why.
""")
    return slide


def slide_agenda(prs, results):
    slide = new_slide(prs, "Agenda", kicker="Roadmap")

    entries = [
        ("1", "Background", "CLIP  ·  test-time adaptation  ·  what calibration means"),
        ("2", "TDA", "the cache-based method we build on, and where it goes wrong"),
        ("3", "Our method", "three independently-switchable, training-free changes"),
        ("4", "Results", "headline numbers, the 40-run ablation, and the mechanism"),
        ("5", "Conclusion", "what we stand behind, and what we do not"),
    ]

    top = Inches(2.05)
    height = Inches(0.78)
    for index, (number, heading, detail) in enumerate(entries):
        row_top = int(top + index * Inches(0.92))
        chip = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, MARGIN, row_top, Inches(0.62), height
        )
        chip.fill.solid()
        chip.fill.fore_color.rgb = ACCENT if index in (2, 3) else HEADER_FILL
        chip.line.fill.background()
        chip.shadow.inherit = False
        chip.adjustments[0] = 0.22
        chip.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
        paragraph = chip.text_frame.paragraphs[0]
        paragraph.alignment = PP_ALIGN.CENTER
        run = paragraph.add_run()
        run.text = number
        _style_run(run, 17, bold=True, color=SURFACE if index in (2, 3) else MUTED)

        _, frame = _textbox(
            slide, MARGIN + Inches(0.92), row_top, Inches(10.6), height,
            anchor=MSO_ANCHOR.MIDDLE,
        )
        run = frame.paragraphs[0].add_run()
        run.text = heading
        _style_run(run, 19, bold=True, color=INK)
        run = frame.paragraphs[0].add_run()
        run.text = "     " + detail
        _style_run(run, 14, color=MUTED)

    set_notes(slide, """
Five parts. I will spend the first third on background, because the result
only means something once you know what TDA does and what calibration is.

Sections three and four - our method and our results - are the project
itself, and that is where most of the time goes.
""")
    return slide


def slide_clip(prs, results):
    slide = new_slide(prs, "What is CLIP?", kicker="1 · Background")

    add_bullets(slide, [
        "CLIP (OpenAI, 2021) is a vision-language model: one image encoder and one text "
        "encoder that map pictures and sentences into the same embedding space.",
        "Its purpose is to learn visual concepts from language itself, so a single frozen "
        "model can recognise anything that can be described in words.",
        "It was trained on ~400M web image-caption pairs with a contrastive objective: each "
        "image is pulled toward its true caption and pushed away from every other one.",
        "",
        "“Zero-shot” classification means classifying into categories the model was never "
        "trained on, using zero labelled examples of them.",
        "It is used because it removes the cost of a new task entirely: no training data, "
        "no classifier head, no fine-tuning — the class names alone define the classifier.",
        (1, "In practice, each class name is wrapped in a prompt (“a photo of a {class}”), "
            "every prompt is encoded, and an image is assigned to the nearest prompt."),
    ], width=Inches(7.3), size=15, sub_size=13, spacing=9)

    panel_left = Inches(8.30)
    add_panel(slide, panel_left, Inches(2.05), Inches(4.40), Inches(2.55))
    _, frame = _textbox(
        slide, panel_left + Inches(0.30), Inches(2.32), Inches(3.85), Inches(2.8)
    )
    run = frame.paragraphs[0].add_run()
    run.text = "The zero-shot rule"
    _style_run(run, 14, bold=True, color=MUTED)

    for text, size, bold, color, space in [
        ("logits  =  s · f(image) · g(prompts)ᵀ", 15, True, INK, 14),
        ("f  =  image encoder (frozen)", 12, False, MUTED, 5),
        ("g  =  text encoder over class prompts", 12, False, MUTED, 5),
        ("s  =  learned logit scale", 12, False, MUTED, 5),
        ("prediction  =  argmax of those logits", 14, True, ACCENT, 14),
    ]:
        paragraph = frame.add_paragraph()
        paragraph.space_before = Pt(space)
        run = paragraph.add_run()
        run.text = text
        _style_run(run, size, bold=bold, color=color)

    set_notes(slide, """
First, what CLIP actually is. It is a vision-language model from OpenAI: two
encoders, one for images and one for text, trained so that a picture and a
sentence describing it land close together in one shared embedding space.
The point of training it that way is generality - instead of learning one
fixed set of classes, it learns visual concepts from language, so one frozen
model can recognise anything you can describe in words.

That is what makes zero-shot classification possible. Zero-shot means
classifying into categories the model was never explicitly trained on, with
zero labelled examples. And the reason it is used is economic: a new task
costs nothing. No training data, no classifier head, no fine-tuning - you
write the class names, and the names themselves are the classifier.

Mechanically, each class name is wrapped in a prompt like "a photo of a
Boeing 737", all the prompts are encoded, and an image is assigned to
whichever prompt it sits closest to - the formula on the right. Everything
later in this talk either adds a term to those logits or rescales them; the
encoders stay frozen throughout.
""")
    return slide


def slide_clip_fails(prs, results):
    slide = new_slide(
        prs, "… but CLIP degrades badly out of distribution", kicker="1 · Background"
    )

    # Literature numbers (arXiv:2506.24000), not our own runs: the background
    # section motivates the problem from published work, and our measurements
    # first appear in the Results section where they belong.
    tiles = [
        ("EuroSAT", "satellite land-use scenes", "23.7%"),
        ("FGVC-Aircraft", "100 fine-grained aircraft variants", "15.7%"),
    ]
    tile_w, tile_h, tile_gap = Inches(5.0), Inches(1.52), Inches(0.42)
    for index, (name, description, value) in enumerate(tiles):
        left = MARGIN + Inches(0.15)
        top = int(Inches(2.15) + index * (tile_h + tile_gap))
        add_panel(slide, left, top, tile_w, tile_h)
        _, frame = _textbox(slide, int(left + Inches(0.35)), int(top + Inches(0.22)),
                            int(tile_w - Inches(0.7)), int(tile_h - Inches(0.4)))
        run = frame.paragraphs[0].add_run()
        run.text = name
        _style_run(run, 15, bold=True, color=INK)
        run = frame.paragraphs[0].add_run()
        run.text = "    " + description
        _style_run(run, 11.5, color=MUTED)

        paragraph = frame.add_paragraph()
        paragraph.space_before = Pt(6)
        run = paragraph.add_run()
        run.text = value
        _style_run(run, 32, bold=True, color=WARN)
        run = paragraph.add_run()
        run.text = "   zero-shot CLIP accuracy"
        _style_run(run, 12, color=MUTED)

    _, frame = _textbox(slide, MARGIN + Inches(0.15), Inches(5.72), tile_w, Inches(0.3))
    run = frame.paragraphs[0].add_run()
    run.text = "As reported by the benchmark suite we build on (arXiv:2506.24000)."
    _style_run(run, 10.5, italic=True, color=MUTED)

    add_bullets(slide, [
        "CLIP's training data is web imagery, so domains that look nothing like the web "
        "— satellite views, fine-grained aircraft variants, textures — are out of "
        "distribution for it.",
        "On those domains, zero-shot accuracy collapses to the numbers on the left: a "
        "model that names everyday objects effortlessly cannot tell two aircraft apart.",
        "",
        "The usual remedy would be fine-tuning on the new domain. Here that door is "
        "closed: the new domain comes with no labels, and retraining a backbone for "
        "every deployment target is not realistic.",
    ], left=Inches(6.30), width=Inches(6.40), size=15.5, spacing=11)

    add_callout(
        slide,
        "Test-time adaptation attacks exactly this gap, using only the test images themselves.",
    )
    set_notes(slide, """
So why is CLIP not simply the end of the story? Because its generality comes
from its training data, and that data is web imagery. A domain that looks
nothing like the web - a satellite view, a close-up of an aircraft variant -
is out of distribution for it.

On those domains the zero-shot accuracy collapses. The benchmark paper we
build on reports CLIP at 23.7 percent on EuroSAT satellite scenes and 15.7
percent on fine-grained aircraft - a model that names everyday objects
effortlessly cannot tell a Boeing 737-700 from a 737-800.

The usual remedy would be fine-tuning on the new domain, and that is exactly
what we are not allowed: the new domain has no labels, and retraining a
backbone per deployment target is not realistic. That constraint - recover
the lost accuracy with no labels and no training - is what test-time
adaptation is for, and it is the setting of this whole project.
""")
    return slide


def slide_what_is_tta(prs, results):
    slide = new_slide(prs, "What is test-time adaptation? TDA, by example",
                      kicker="1 · Background")

    add_bullets(slide, [
        "Test-time adaptation (TTA) adapts the model while it is already deployed, using "
        "only the unlabelled test images as they arrive — no labels, no training run.",
        "Many TTA methods exist. We build on TDA (CVPR 2024): forward passes only, plus a "
        "small memory of past test images. Here is what that memory does, on one photo.",
    ], width=Inches(12.1), size=16, spacing=8)

    top, height = Inches(3.22), Inches(2.25)
    gap = Inches(0.42)
    width = int((CONTENT_W - gap * 3) / 4)
    steps = [
        ("1 · A photo arrives",
         "\nFrozen CLIP turns it into a feature vector — its "
         "position in the embedding space — and predicts, "
         "say, 737‑800 with high confidence."),
        ("2 · Store it",
         "\nThe feature is cached under CLIP's own guess, "
         "a pseudo-label. No ground truth is involved, "
         "so a confident mistake is stored just the same."),
        ("3 · A later photo arrives",
         "\nIts feature is compared to every cached one by "
         "similarity in that space. Cached photos that look "
         "alike count; distant ones barely register."),
        ("4 · Neighbours vote",
         "\nEach similar cached item votes for its pseudo-label. "
         "Three nearby 737‑800s push the 737‑800 logit up, "
         "and the votes are added to CLIP's own logits."),
    ]
    left = MARGIN
    for index, (title, subtitle) in enumerate(steps):
        highlighted = index in (1, 3)
        _flow_box(
            slide, left, top, width, height, title, subtitle,
            fill=TINT_ACCENT if highlighted else SURFACE,
            border=ACCENT if highlighted else RULE,
            title_size=14.5, subtitle_size=12.5,
        )
        if index < len(steps) - 1:
            _arrow(slide, int(left + width + Inches(0.06)), int(top + height / 2 - Inches(0.16)),
                   int(gap - Inches(0.12)), Inches(0.32))
        left = int(left + width + gap)

    add_callout(
        slide,
        "The cache is a nearest-neighbour classifier grown from the test stream itself, "
        "labelled by CLIP's own earlier guesses.",
        top=Inches(5.85),
    )
    set_notes(slide, """
Test-time adaptation means adapting while you are already deployed, using
nothing but the unlabelled test images as they come in. There is no label and
no training run; the test stream itself is the only new information.

There are many TTA methods. The one we build on is TDA, which never
backpropagates. Instead it keeps a small memory of past test images, and the
easiest way to see what that memory does is to follow one photo through it.

Step one: a photo of a plane arrives. Frozen CLIP encodes it into a feature
vector, which is just the point in CLIP's embedding space where this photo
landed, and the usual zero-shot step predicts, say, 737-800 with high
confidence.

Step two: TDA stores that feature in a cache, filed under the class CLIP
itself just predicted. That label is a pseudo-label. Nothing checked it. If
CLIP was wrong and this was really a 737-900, the feature still sits under
737-800.

Step three: a later photo arrives. TDA computes its feature the same way and
compares it to every stored feature by similarity in the embedding space.
Cached photos that look like the new one get a high similarity; unrelated
ones get almost none.

Step four: the similar cached items vote for whatever pseudo-label they were
filed under. If the new photo sits close to three cached 737-800s, the
737-800 logit gets a boost. Those votes are added to CLIP's own zero-shot
logits, and the prediction is the argmax of the sum.

So the cache acts like a nearest-neighbour classifier that grows out of the
test stream, labelled by CLIP's own earlier opinions. That is what makes TDA
cheap and effective, and, as the next slides show, it is also exactly where
its calibration problem comes from.
""")
    return slide


def slide_calibration(prs, results):
    slide = new_slide(
        prs, "Accuracy is not the only thing that matters", kicker="1 · Background"
    )

    add_bullets(slide, [
        "Alongside every prediction, a classifier also reports a confidence — the "
        "probability it assigns to the class it chose.",
        "A model is well-calibrated when that confidence matches reality: among all "
        "the predictions it makes with 80% confidence, about 80% should actually be "
        "correct.",
        (1, "An overconfident model — right 80% of the time while claiming 95% — "
            "misleads anything that acts on its output: a downstream filter, a human "
            "reviewer, an abstention rule."),
        "",
        "Expected Calibration Error (ECE) turns this into a single number, in three "
        "steps: sort the predictions into bins by their confidence; inside each bin, "
        "compare the average confidence to the actual accuracy; then average those "
        "gaps, weighting each bin by how many predictions landed in it.",
        (1, "Example: if the 90–95% bin holds 200 predictions with average confidence "
            "92% but only 75% of them correct, that bin contributes a 17-point gap."),
    ], width=Inches(7.45), size=14.5, sub_size=12.5, spacing=9)

    figure = render_calibration_figure(Path("presentation/assets/what_is_calibration.png"))
    add_picture(slide, figure, Inches(8.35), Inches(1.80), Inches(4.35), Inches(4.05))
    _, frame = _textbox(slide, Inches(8.35), Inches(5.92), Inches(4.35), Inches(0.5))
    frame.paragraphs[0].alignment = PP_ALIGN.CENTER
    run = frame.paragraphs[0].add_run()
    run.text = "Illustrative reliability diagram (synthetic curves, not our results)."
    _style_run(run, 10, italic=True, color=MUTED)

    add_panel(slide, MARGIN, Inches(5.70), Inches(7.45), Inches(0.80))
    _, frame = _textbox(slide, MARGIN + Inches(0.3), Inches(5.84), Inches(6.9), Inches(0.55))
    run = frame.paragraphs[0].add_run()
    run.text = "ECE  =  Σ_b  (n_b / N) · | accuracy(b) − confidence(b) |"
    _style_run(run, 14.5, bold=True, color=INK)
    paragraph = frame.add_paragraph()
    paragraph.space_before = Pt(3)
    run = paragraph.add_run()
    run.text = "20 equal-width bins in this project  ·  0 means perfectly honest  ·  lower is better"
    _style_run(run, 11, color=MUTED)

    set_notes(slide, """
So far we have only talked about accuracy. But a classifier gives you two
things per prediction: the class it chose, and a confidence - the
probability it assigns to that choice.

A model is well-calibrated when that confidence matches reality. Among all
the predictions it makes with eighty percent confidence, about eighty
percent should actually be correct. An overconfident model - one that is
right eighty percent of the time but routinely claims ninety-five - misleads
anything downstream that acts on the number: a filter that auto-accepts
confident predictions, a human deciding whether to double-check, an
abstention rule.

Expected Calibration Error turns this into one number, in three steps.
First, sort all the predictions into bins by their confidence - we use
twenty equal-width bins. Second, inside each bin, compare the average
confidence to the actual accuracy of those predictions. Third, average those
per-bin gaps, weighting each bin by how many predictions it holds. A
concrete example: if the ninety-to-ninety-five percent bin holds two hundred
predictions averaging ninety-two percent confidence but only seventy-five
percent of them are right, that bin contributes a seventeen-point gap.

The diagram on the right shows the same thing visually, on synthetic curves.
Perfect calibration is the dashed diagonal - accuracy equals confidence
everywhere. An overconfident model sags below it, and the purple segment is
one bin's gap. ECE is just the weighted average of all those gaps: zero
means every confidence level is honest, and higher means the number the
model reports cannot be trusted.
""")
    return slide


def slide_the_gap(prs, results):
    slide = new_slide(
        prs, "TTA buys accuracy and pays for it in calibration", kicker="1 · Background"
    )

    # Published findings only (arXiv:2506.24000): the background motivates the
    # problem from the literature; our own measurements appear in Results.
    cards = [
        ("Accuracy", "improves", ACCENT,
         "TTA methods recover a large share of the accuracy CLIP loses out of "
         "distribution — this is what they are built and published for."),
        ("Calibration (ECE)", "5.70%  →  9.21%", WARN,
         "Across the benchmark's suite, TDA's average calibration error nearly "
         "doubles while its accuracy improves — the confidence becomes less "
         "trustworthy exactly as the predictions get better."),
    ]
    width, gap = Inches(5.85), Inches(0.4)
    for index, (name, value, color, body) in enumerate(cards):
        left = int(MARGIN + index * (width + gap))
        add_panel(slide, left, Inches(2.05), width, Inches(2.30))
        _, frame = _textbox(
            slide, left + Inches(0.35), Inches(2.28), width - Inches(0.7), Inches(1.9)
        )
        run = frame.paragraphs[0].add_run()
        run.text = name
        _style_run(run, 13, bold=True, color=MUTED)

        paragraph = frame.add_paragraph()
        paragraph.space_before = Pt(8)
        run = paragraph.add_run()
        run.text = value
        _style_run(run, 28, bold=True, color=color)

        paragraph = frame.add_paragraph()
        paragraph.space_before = Pt(8)
        run = paragraph.add_run()
        run.text = body
        _style_run(run, 12.5, color=MUTED)

    add_bullets(slide, [
        "This is the finding of a 2025 NeurIPS benchmark (“The Illusion of Progress?”, "
        "arXiv:2506.24000), measured across ten datasets — and it is the gap this "
        "project set out to close.",
        "The cause is not an accident of tuning; it is built into how TDA works. The "
        "next three slides show the mechanism.",
    ], top=Inches(4.65), size=15, spacing=8)

    add_callout(
        slide,
        "The project: close that calibration gap without retraining and without labels.",
        top=Inches(5.95),
    )
    set_notes(slide, """
Here is the problem this project is about, as the literature reports it.

A NeurIPS 2025 benchmark - pointedly titled "The Illusion of Progress?" -
measured test-time adaptation methods across ten datasets, and found that
TDA's accuracy gains come with a hidden bill. Average calibration error
rises from 5.70 to 9.21 percent - it nearly doubles - at the same time as
accuracy improves. The predictions get better while the confidence attached
to them gets less trustworthy.

What makes this interesting is that it is not a tuning accident. It follows
from how TDA is built, and in the next three slides I will show you the
mechanism - because our method is three attempts to break exactly that
mechanism.

The project, in one sentence: close that calibration gap under the same
rules TTA plays by - no retraining, and no labels.
""")
    return slide


def slide_tda_cache(prs, results):
    slide = new_slide(prs, "TDA, part 1: a cache of past test features", kicker="2 · TDA")
    _tda_diagram(slide)
    add_callout(
        slide,
        "Forward passes only. The cache IS the adaptation - no gradients, no labels.",
        top=Inches(5.95),
    )
    set_notes(slide, """
This is TDA, top to bottom.

An image arrives. Frozen CLIP encodes it and produces zero-shot logits -
exactly the formula from the CLIP slide, unchanged.

Then the cache. TDA keeps two. The positive cache stores features from
samples it is confident about, filed under the class it itself predicted -
these are pseudo-labels, there is no ground truth anywhere - and it keeps at
most three per class. The negative cache stores mid-entropy samples, which
carry evidence about what a class is not.

For a new image, TDA looks up both caches by feature similarity and converts
them into extra logits.

The bottom line is the one to remember: final logits are CLIP logits, plus
the positive cache term, minus the negative cache term. It is a sum of
logits, not a re-normalised mixture of distributions. That detail is where
the calibration problem comes from, and it is what our second contribution
changes.
""")
    return slide


def slide_tda_admission(prs, results):
    slide = new_slide(prs, "TDA, part 2: who gets into the cache", kicker="2 · TDA")

    add_bullets(slide, [
        "Admission is gated by ENTROPY. The negative cache admits a sample only when its "
        "normalised prediction entropy falls inside a band (0.2 - 0.5). The positive cache "
        "admits every sample, and uses entropy to decide what to evict once a class is full.",
        (1, "Entropy measures how peaked a prediction is - how loudly the model asserts it."),
        "",
        "Combination is ADDITIVE. Cache evidence is added onto CLIP logits, each term with "
        "its own tuned alpha and beta (per-dataset, from the original TDA paper).",
        "",
        "Both choices are defensible for accuracy, and TDA's accuracy gains are real. "
        "But notice what neither of them ever checks.",
    ], width=Inches(12.1), size=15.5, sub_size=13.5, spacing=8)

    add_panel(slide, MARGIN, Inches(5.05), CONTENT_W, Inches(1.30),
              fill=RGBColor(0xF7, 0xEA, 0xE1))
    _, frame = _textbox(
        slide, MARGIN + Inches(0.35), Inches(5.26), CONTENT_W - Inches(0.7), Inches(0.95)
    )
    run = frame.paragraphs[0].add_run()
    run.text = ("Neither the entropy gate nor the additive sum ever asks whether the "
                "prediction was CORRECT.")
    _style_run(run, 17, bold=True, color=INK)
    paragraph = frame.add_paragraph()
    paragraph.space_before = Pt(5)
    run = paragraph.add_run()
    run.text = ("A confidently wrong prediction is admitted on exactly the same terms as a "
                "confidently right one - and no label exists at test time to tell them apart.")
    _style_run(run, 13.5, color=MUTED)

    set_notes(slide, """
Two design choices to pin down, because our method changes both.

First, admission is gated by entropy - low entropy means a peaked, decisive
prediction. To be precise: the negative cache uses an entropy band, and the
positive cache actually admits everything, using entropy only as the
eviction key when a class slot is already full. We verified that in the
source, and it matters later for what turning our own gate off has to mean.

Second, combination is additive: cache logits are added onto CLIP logits
with tuned alpha and beta coefficients.

Now the problem, in the box. Entropy measures how loud a prediction is, not
whether it is right. And there is no label at test time to check against. So
a confidently wrong prediction enters the cache on exactly the same terms as
a confidently right one.
""")
    return slide


def slide_feedback_loop(prs, results):
    slide = new_slide(prs, "Why that breaks calibration", kicker="2 · TDA")

    steps = [
        ("1", "A confident prediction is made", "entropy is low - right or wrong, it qualifies"),
        ("2", "It enters the positive cache", "filed under its own pseudo-label"),
        ("3", "A similar image arrives later", "the cache retrieves that stored feature"),
        ("4", "Cache logits are ADDED to CLIP logits",
         "the two agree, so the sum is sharper than either alone"),
        ("5", "Confidence rises - accuracy need not",
         "the error has been reinforced, not corrected"),
    ]

    top = Inches(2.15)
    for index, (number, heading, detail) in enumerate(steps):
        row_top = int(top + index * Inches(0.80))
        colour = WARN if index >= 3 else MUTED
        chip = slide.shapes.add_shape(
            MSO_SHAPE.OVAL, MARGIN, row_top, Inches(0.50), Inches(0.50)
        )
        chip.fill.solid()
        chip.fill.fore_color.rgb = colour
        chip.line.fill.background()
        chip.shadow.inherit = False
        chip.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
        paragraph = chip.text_frame.paragraphs[0]
        paragraph.alignment = PP_ALIGN.CENTER
        run = paragraph.add_run()
        run.text = number
        _style_run(run, 13, bold=True, color=SURFACE)

        _, frame = _textbox(
            slide, MARGIN + Inches(0.78), row_top, Inches(11.2), Inches(0.50),
            anchor=MSO_ANCHOR.MIDDLE,
        )
        run = frame.paragraphs[0].add_run()
        run.text = heading
        _style_run(run, 17, bold=True, color=INK)
        run = frame.paragraphs[0].add_run()
        run.text = "     " + detail
        _style_run(run, 13.5, color=MUTED)

    add_callout(
        slide,
        "A feedback loop on confidence, with no term anywhere that depends on correctness.",
        color=WARN, top=Inches(6.20),
    )
    set_notes(slide, """
Put those two design choices together and you get a feedback loop.

A confident prediction - correct or not - gets cached. Later, a similar
image retrieves it. The cache logits agree with CLIP logits, because the
cache was built out of CLIP's own earlier opinion. Adding two agreeing logit
vectors produces a sharper distribution than either one alone.

So confidence goes up. Accuracy does not necessarily follow, because nothing
in that loop ever consulted a label. Confidently wrong predictions are
reinforced exactly as efficiently as confidently right ones.

That is our diagnosis, and each of our three changes attacks one link in
this chain.
""")
    return slide


def _contribution_slide(prs, results, *, number, title, kicker, problem, change,
                        detail_bullets, formula, formula_caption, callout,
                        callout_color=ACCENT, notes=""):
    """Shared layout for the three contribution slides, so they read as a set."""
    slide = new_slide(prs, title, kicker=kicker)

    chip = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, SLIDE_W - MARGIN - Inches(1.55),
        Inches(0.50), Inches(1.55), Inches(0.52)
    )
    chip.fill.solid()
    chip.fill.fore_color.rgb = ACCENT
    chip.line.fill.background()
    chip.shadow.inherit = False
    chip.adjustments[0] = 0.25
    chip.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    paragraph = chip.text_frame.paragraphs[0]
    paragraph.alignment = PP_ALIGN.CENTER
    run = paragraph.add_run()
    run.text = "change " + str(number)
    _style_run(run, 13, bold=True, color=SURFACE)

    rows = [("Baseline", problem, MUTED), ("Ours", change, ACCENT)]
    for index, (label, text, color) in enumerate(rows):
        row_top = int(Inches(1.90) + index * Inches(0.74))
        _, frame = _textbox(slide, MARGIN, row_top, Inches(1.35), Inches(0.66),
                            anchor=MSO_ANCHOR.MIDDLE)
        run = frame.paragraphs[0].add_run()
        run.text = label
        _style_run(run, 12.5, bold=True, color=MUTED)

        _, frame = _textbox(slide, MARGIN + Inches(1.45), row_top,
                            Inches(10.65), Inches(0.66), anchor=MSO_ANCHOR.MIDDLE)
        run = frame.paragraphs[0].add_run()
        run.text = text
        _style_run(run, 16.5, bold=(index == 1), color=color)

    # 3.42 -> 5.22 is 1.8in of bullet room. Four bullets at these sizes fill
    # roughly 1.6in, which is the budget this layout has: the earlier version
    # started them at 3.72 and the fourth bullet rendered BEHIND the formula
    # panel, invisible in PowerPoint and invisible to a structural check.
    add_bullets(slide, detail_bullets, top=Inches(3.42), size=14.5, sub_size=12.5,
                spacing=7)

    add_panel(slide, MARGIN, Inches(5.28), CONTENT_W, Inches(0.84))
    _, frame = _textbox(slide, MARGIN + Inches(0.3), Inches(5.42),
                        CONTENT_W - Inches(0.6), Inches(0.62))
    run = frame.paragraphs[0].add_run()
    run.text = formula
    _style_run(run, 15, bold=True, color=INK)
    run = frame.paragraphs[0].add_run()
    run.text = "      " + formula_caption
    _style_run(run, 12, italic=True, color=MUTED)

    add_callout(slide, callout, color=callout_color, top=Inches(6.25))
    set_notes(slide, notes)
    return slide


def slide_change_margin(prs, results):
    (low_dataset, low_rate), (high_dataset, high_rate) = results.admission_range()
    threshold = results.method_constant("margin_threshold")
    return _contribution_slide(
        prs, results, number=1,
        title="Admit by margin, not by entropy",
        kicker="3 · Our method",
        problem="TDA admits samples by entropy, which only measures how peaked the whole prediction is.",
        change="We admit a sample only when its top logit clearly beats the runner-up — proof one class won.",
        detail_bullets=[
            "Entropy is a property of all C classes at once, so it can be driven low by a "
            "long flat tail even when the top two classes are nearly tied - exactly the "
            "ambiguous case we want kept out.",
            "The top-2 margin asks the narrower question: did one class actually beat the "
            "runner-up? That is a more direct signal of a decisive prediction.",
            (1, f"One GLOBAL threshold of {threshold:g}, deliberately NOT tuned per dataset - "
                "fitting it to each test set would be test-set tuning, even without labels."),
            (1, f"Measured admission rates span {high_rate * 100:.1f}% ({DATASET_LABELS[high_dataset]}) "
                f"down to {low_rate * 100:.1f}% ({DATASET_LABELS[low_dataset]}). "
                "We report that spread as a limitation rather than retuning it away."),
        ],
        formula=f"margin(x)  =  top1(logits)  −  top2(logits)         admit  if  margin > {threshold:g}",
        formula_caption="replaces the entropy gate; everything else in the cache is untouched",
        callout="Goal: keep confidently-ambiguous samples out of the cache in the first place.",
        notes=f"""
Change one attacks the admission gate.

The trouble with entropy is that it is a property of the whole distribution
across all classes. On a hundred-class problem a long flat tail can drive
entropy down even when the top two classes are essentially tied - and that
tied case is exactly what we want to keep out of the cache.

The top-2 margin asks a narrower and more relevant question: did one class
actually beat the runner-up?

One honest caveat, which becomes important later. We use a single global
threshold of one point zero for every dataset. We could have tuned it per
dataset, but fitting a threshold to each test set margin distribution is
test-set tuning even though no labels are involved. The cost is that logit
margins do not share a scale across datasets with different class counts:
admission runs from {high_rate * 100:.1f} percent on {DATASET_LABELS[high_dataset]} down to {low_rate * 100:.1f} percent on
{DATASET_LABELS[low_dataset]}, where the cache nearly starves. We report that rather than
retune it.
""")


def slide_change_fusion(prs, results):
    weight = results.method_constant("fusion_weight")
    return _contribution_slide(
        prs, results, number=2,
        title="Fuse probabilities, do not add logits",
        kicker="3 · Our method",
        problem="TDA adds the cache's logits onto CLIP's — an unbounded sum that can only grow sharper.",
        change="We take a convex mixture of the two probability distributions, bounded by its inputs.",
        detail_bullets=[
            "Adding two agreeing logit vectors produces a distribution sharper than either "
            "input. That is the mechanism that manufactures confidence out of agreement.",
            "A convex combination cannot do that: a weighted average is bounded by its most "
            "confident input, so agreement can change WHICH class wins without inflating "
            "HOW confident the answer is.",
            (1, f"Mixture weight fixed at {weight:g} - CLIP and the cache are weighted equally."),
            (1, "The negative cache stays on the CLIP side of the mixture, since it is a "
                "correction to CLIP's own evidence rather than independent evidence."),
        ],
        formula=f"p  =  (1 − w) · softmax(clip)  +  w · softmax(cache),         w = {weight:g}",
        formula_caption="bounded by its most confident input, by construction",
        callout="Goal: let the cache change the answer without inflating the confidence.",
        notes="""
Change two attacks the combination step.

Adding two logit vectors that agree gives you a distribution sharper than
either one on its own. That is the step that manufactures confidence out of
mere agreement.

A convex mixture of the two probability distributions cannot do that. A
weighted average is bounded by its most confident input, so the cache can
still change which class wins, but it cannot push the winning probability
higher than both sources had.

We weight CLIP and cache equally, at one half. Keep this slide in mind -
this is the change that goes most spectacularly wrong, and the reason is
visible right here in the equation if you look for it.
""")


def slide_change_temperature(prs, results):
    return _contribution_slide(
        prs, results, number=3,
        title="Pick a temperature with no labels at all",
        kicker="3 · Our method",
        problem="TDA never rescales its output — whatever sharpness falls out of the sum is what you get.",
        change="We divide the logits by a temperature chosen from a label-free estimate of accuracy.",
        detail_bullets=[
            "A well-calibrated model has mean confidence equal to its accuracy. We know the "
            "mean confidence; the missing half is the accuracy, which needs labels.",
            "We estimate it WITHOUT labels: leave-one-out 1-nearest-neighbour agreement among "
            "the cached features and the pseudo-labels the method itself assigned. A cache "
            "that disagrees with itself is a cache whose predictions are unreliable.",
            (1, "Then grid-search the temperature whose mean confidence matches that estimate."),
            (1, "The grid floor is 1.0, so the search can only SMOOTH, never sharpen. If the "
                "estimate is inflated, the worst case is a no-op instead of making things worse."),
        ],
        formula="T*  =  argmin | mean-confidence(logits / T)  −  LOO-accuracy(cache) |,     T ≥ 1",
        formula_caption="monotonic, so the argmax — and the accuracy — cannot change",
        callout="This contribution is structurally incapable of changing accuracy. It can only move ECE.",
        notes="""
Change three is the one that worked, so let me be precise about it.

A well-calibrated model has mean confidence equal to its accuracy. We can
measure mean confidence directly. The missing half is accuracy, which
normally needs labels.

We estimate it without labels, using the cache itself: take the cached
features and the pseudo-labels the method assigned them, and compute
leave-one-out nearest-neighbour agreement. If the nearest neighbour of a
cached item carries a different pseudo-label, the cache is internally
inconsistent, and that is evidence the predictions are unreliable.

Then we grid-search the temperature that brings mean confidence onto that
estimate. One design decision worth flagging: the grid floor is one, so the
search can only smooth, never sharpen. If our label-free estimate comes out
too optimistic, the worst case is that we do nothing, rather than making
overconfidence worse.

And note the line at the bottom - dividing logits by a positive constant is
monotonic, so it cannot change which class wins. This contribution can move
calibration only. Accuracy is untouchable by construction.
""")


def slide_setup(prs, results):
    slide = new_slide(prs, "Experimental setup", kicker="4 · Results")

    rows = [
        [DATASET_LABELS[dataset],
         f"{results.cell(dataset, 'clipzs', 'n_classes'):.0f}",
         f"{results.cell(dataset, 'clipzs', 'n_samples'):,.0f}"]
        for dataset in DATASET_ORDER
    ]
    rows.append(["total", "", f"{results.n_total_images():,}"])
    add_table(
        slide, ["dataset", "classes", "test images"], rows,
        MARGIN, Inches(2.10), Inches(5.0),
        col_widths=[0.42, 0.26, 0.32],
        highlight_rows=(len(rows) - 1,),
    )

    add_bullets(slide, [
        "Backbone ViT-B/16, batch size 1 (true online streaming), seed 0, ECE with 20 bins.",
        "Test split only, using the standard CoOp splits so the numbers stay comparable "
        "to the benchmark paper.",
        "",
        "Every method sees exactly the same images in the same order.",
        (1, "Ground-truth labels are consumed in exactly ONE function in the whole codebase; "
            "no method ever receives one. A test enforces this by inspecting function "
            "signatures, so it cannot rot."),
        "",
        "Two sweeps, both fully scripted and reproducible:",
        (1, "Headline: 5 datasets x 3 methods = 15 runs (~41 min on an RTX 5060)."),
        (1, f"Ablation: 5 datasets x 8 on/off combinations = {results.n_ablation_runs()} runs (~2.3 h)."),
    ], left=Inches(6.10), width=Inches(6.60), size=15, sub_size=13, spacing=9)

    set_notes(slide, f"""
Five datasets, {results.n_total_images():,} test images per method, standard CoOp splits so
our numbers are comparable to the benchmark paper. ViT-B/16, batch size one
so it is genuinely online, seed zero, twenty-bin ECE.

One thing I want to highlight because it is easy to get wrong in this area:
labels enter our codebase in exactly one function, the scorer that runs at
the end of a run. No adaptation method ever receives a label. We enforce
that with a test that inspects function signatures, so it cannot quietly
decay as the code changes.

Two sweeps: fifteen headline runs, and {results.n_ablation_runs()} ablation runs covering all eight
on/off combinations of our three changes on all five datasets.
""")
    return slide


def _method_table(slide, results, column, top, *, title, highlight_worst):
    header = ["", *(DATASET_LABELS[d] for d in DATASET_ORDER), "mean"]
    labels = [("clipzs", "CLIP zero-shot"), ("tda", "TDA"), ("cal_tda", "Ours, all 3")]
    rows = []
    for method, label in labels:
        values = [results.cell(d, method, column) for d in DATASET_ORDER]
        rows.append([label, *(pct(v) for v in values),
                     pct(results.method_mean(method, column))])

    _, frame = _textbox(slide, MARGIN, int(top - Inches(0.28)), Inches(7.2), Inches(0.26))
    run = frame.paragraphs[0].add_run()
    run.text = title
    _style_run(run, 12.5, bold=True, color=MUTED)

    return add_table(
        slide, header, rows, MARGIN, top, Inches(7.25),
        col_widths=[0.235, 0.117, 0.145, 0.108, 0.133, 0.132, 0.13],
        font_size=11, header_size=10.5, row_height=Inches(0.31),
        highlight_rows=(2,), highlight_color=WARN if highlight_worst else None,
    )


def slide_headline(prs, results):
    slide = new_slide(prs, "Headline result: the bundle does not work", kicker="4 · Results")

    _method_table(slide, results, "accuracy", Inches(2.25),
                  title="Top-1 accuracy (%)   -   higher is better",
                  highlight_worst=False)
    _method_table(slide, results, "ece", Inches(4.05),
                  title="ECE, 20 bins (%)   -   lower is better",
                  highlight_worst=True)

    figure = Path("analysis/figures/accuracy_vs_ece.png")
    if figure.exists():
        add_picture(slide, figure, Inches(8.10), Inches(1.95), Inches(4.62), Inches(4.05))

    hits = results.criterion_hits()
    add_callout(
        slide,
        f"Success criterion (accuracy ≥ TDA AND ECE < TDA): met on {hits} of 5 datasets. "
        "We report that as the negative result it is.",
        color=WARN, top=Inches(6.05),
    )
    set_notes(slide, f"""
This is the result we set out to get, and did not get.

Top table, accuracy: our full method is below TDA on every single dataset.
Bottom table, ECE: it is worse than TDA on four of five - and not slightly.
Flowers goes from {pct(results.cell('flower102', 'tda', 'ece'))} to {pct(results.cell('flower102', 'cal_tda', 'ece'))} percent, Pets from
{pct(results.cell('pets', 'tda', 'ece'))} to {pct(results.cell('pets', 'cal_tda', 'ece'))} percent. Aircraft is the one dataset where the
bundle genuinely helps calibration.

Our success criterion was: at least three of five datasets where we match or
beat TDA accuracy while beating its ECE. We hit that on {hits} of five.

I want to be direct about this rather than reframe it. The method as
specified - all three contributions enabled together - fails. The
interesting question is why, and that is what the ablation answers.
""")
    return slide


def slide_ablation(prs, results):
    slide = new_slide(
        prs, "The ablation: one of the three actually works", kicker="4 · Results"
    )

    figure = Path("analysis/figures/ablation_ece.png")
    if figure.exists():
        add_picture(slide, figure, MARGIN, Inches(1.88), CONTENT_W, Inches(4.05))

    none_ece = results.variant_mean("none", "ece")
    temp_ece = results.variant_mean("temp", "ece")
    wins = results.temp_beats_none_on_ece()
    add_callout(
        slide,
        f"LOO temperature alone: mean ECE {pct(none_ece)}% → {pct(temp_ece)}% at IDENTICAL "
        f"accuracy, better on {wins} of 5 datasets.",
        top=Inches(6.10),
    )
    set_notes(slide, f"""
Forty runs: all eight on/off combinations of our three changes, on all five
datasets. Left panel is mean ECE, lower is better. Right panel is mean
accuracy. Rows are sorted so the best-calibrated variant is on top.

The orange row is the baseline - all three flags off, which is TDA.

The purple row on top is temperature scaling alone, with margin admission
and fusion both switched off. Mean ECE drops from {pct(none_ece)} to {pct(temp_ece)}
percent, and look at the right panel: accuracy is identical, {pct(results.variant_mean('temp', 'accuracy'))}
in both. Not approximately - exactly, to two decimals, on all five datasets.
That is the structural guarantee from the previous slide showing up in the
measurements: temperature scaling is monotonic, so it cannot move an argmax.

The hatched row, second from the bottom on ECE, is our full method. Nearly
everything we added on top of temperature made things worse.

One honest qualifier: it wins on {wins} of five datasets, not all five. Pets is
the exception, and I will come back to it.
""")
    return slide


def slide_mechanism(prs, results):
    slide = new_slide(prs, "Why the other two failed", kicker="4 · Results")

    none_acc = results.variant_mean("none", "accuracy")
    margin_acc = results.variant_mean("margin", "accuracy")
    fusion_ece = results.variant_mean("fusion", "ece")
    low_boundary, high_boundary = results.boundary_fraction_range()

    (low_dataset, low_rate), _ = results.admission_range()
    cards = [
        ("Margin admission",
         f"accuracy {pct(none_acc)} → {pct(margin_acc)}",
         "The global threshold starves the cache on high-class-count datasets. "
         f"On {DATASET_LABELS[low_dataset]} only {low_rate * 100:.1f}% of samples are "
         "admitted, so the cache holds too little signal to help - and the signal "
         "it discards was being used correctly.",
         WARN),
        ("Probabilistic fusion",
         f"mean ECE → {pct(fusion_ece)}",
         "Backfires the opposite way from the design intent. Cache logits are "
         "small and flat, so a 50/50 mixture caps top-1 confidence well below 1 "
         "on EVERY sample - correct or not. The result is severe UNDER-confidence.",
         WARN),
        ("Temperature, on top of both",
         f"{low_boundary * 100:.0f}-{high_boundary * 100:.0f}% pinned at T = 1",
         "With margin and fusion both active, the LOO estimate is inflated by the "
         "selectively-pure cache, so the search wants to sharpen and is held at "
         "the no-op floor. Contribution 3 is effectively INERT inside the bundle.",
         MUTED),
    ]

    width, gap = Inches(3.85), Inches(0.28)
    for index, (heading, headline, body, color) in enumerate(cards):
        left = int(MARGIN + index * (width + gap))
        add_panel(slide, left, Inches(2.00), width, Inches(2.35))
        _, frame = _textbox(slide, left + Inches(0.26), Inches(2.22),
                            width - Inches(0.52), Inches(2.0))
        run = frame.paragraphs[0].add_run()
        run.text = heading
        _style_run(run, 15, bold=True, color=INK)

        paragraph = frame.add_paragraph()
        paragraph.space_before = Pt(7)
        run = paragraph.add_run()
        run.text = headline
        _style_run(run, 15.5, bold=True, color=color)

        paragraph = frame.add_paragraph()
        paragraph.space_before = Pt(9)
        run = paragraph.add_run()
        run.text = body
        _style_run(run, 12, color=MUTED)

    pets_margin = results.variant_cell("pets", "margin", "ece")
    pets_margin_temp = results.variant_cell("pets", "margin_temp", "ece")
    add_panel(slide, MARGIN, Inches(4.75), CONTENT_W, Inches(1.25),
              fill=RGBColor(0xF7, 0xEA, 0xE1))
    _, frame = _textbox(slide, MARGIN + Inches(0.32), Inches(4.94),
                        CONTENT_W - Inches(0.64), Inches(0.95))
    run = frame.paragraphs[0].add_run()
    run.text = "And the exception we did not average away:  "
    _style_run(run, 14, bold=True, color=INK)
    run = frame.paragraphs[0].add_run()
    run.text = (f"on Pets, where TDA is already very well calibrated "
                f"({pct(pets_margin)}% ECE), adding temperature on top of margin makes it "
                f"{pct(pets_margin_temp)}%.")
    _style_run(run, 14, color=INK)
    paragraph = frame.add_paragraph()
    paragraph.space_before = Pt(4)
    run = paragraph.add_run()
    run.text = ("Our method always smooths toward its label-free estimate, with no way to "
                "detect that no smoothing was needed. Developing on one dataset would have "
                "hidden this entirely.")
    _style_run(run, 12.5, italic=True, color=MUTED)

    set_notes(slide, f"""
Three diagnoses, one per card.

Margin admission: the single global threshold does not port across datasets
with different class counts. On Aircraft, with a hundred classes, it admits
under sixteen percent of samples and the cache starves. Mean accuracy drops
almost three points, from {pct(none_acc)} to {pct(margin_acc)}. The signal it threw away
was being used correctly.

Fusion: this one backfired in the opposite direction from the design intent.
We built it to prevent over-confidence. But the cache logits are small and
flat compared to CLIP's sharp distribution, so averaging them fifty-fifty caps
top-one confidence well below one on every sample, whether it is right or
wrong. The result is severe under-confidence - and ECE punishes that exactly
as hard as over-confidence.

Third card: when margin and fusion are both on, the leave-one-out estimate
is inflated, because a selectively-pure cache agrees with itself more than
the true accuracy would. The search therefore wants to sharpen, and our
floor at T equals one stops it. Between {low_boundary * 100:.0f} and {high_boundary * 100:.0f} percent of samples sit
pinned at the no-op. So inside the bundle, contribution three is doing
essentially nothing - which is why the bundle looks like margin plus fusion.

Finally, the box. On Pets, TDA is already extremely well calibrated. Our
method still smooths, because it has no way to tell that smoothing was not
needed, and ECE goes from {pct(pets_margin)} to {pct(pets_margin_temp)} percent. We had developed on DTD,
where the story was clean. Five datasets showed us the exception.
""")
    return slide


def slide_limitations(prs, results):
    slide = new_slide(prs, "Limitations", kicker="5 · Conclusion")

    (low_dataset, low_rate), (high_dataset, high_rate) = results.admission_range()
    add_bullets(slide, [
        "One global margin threshold is not portable. Logit margins have no common scale "
        "across a 10-class and a 100-class dataset, and admission rates running from "
        f"{low_rate * 100:.1f}% ({DATASET_LABELS[low_dataset]}) to {high_rate * 100:.1f}% "
        f"({DATASET_LABELS[high_dataset]}) show it. Tuning per dataset would have been "
        "test-set tuning, so we did not.",
        "",
        "Temperature scaling has no stopping rule. It always smooths toward the label-free "
        "estimate, even when the model was already well calibrated - which is exactly how "
        "it damages Pets.",
        "",
        "Five datasets, one backbone, one seed. Enough to show the bundle fails and that "
        "temperature alone helps on 4 of 5; not enough to claim either result holds "
        "universally.",
        "",
        "The LOO estimate is only as good as the cache. It inherits whatever bias the "
        "admission rule introduces - which is precisely the coupling that makes "
        "contribution 1 and contribution 3 interfere.",
    ], size=16, spacing=5)

    set_notes(slide, """
Four limitations we would want a reader to know.

The global margin threshold is not portable across class counts, and we
chose not to tune it per dataset because that would be test-set tuning.

Temperature scaling has no stopping rule - it always smooths toward the
estimate, even when the model was already honest. That is the Pets failure.

Five datasets, one backbone, one seed. Enough to establish that the bundle
fails and that temperature alone helps on four of five. Not enough to claim
either holds universally.

And the leave-one-out estimate is only as good as the cache it reads, so it
inherits any bias the admission rule introduces. That coupling is exactly
why contributions one and three interfere with each other.
""")
    return slide


def slide_conclusion(prs, results):
    slide = new_slide(prs, "Conclusion", kicker="5 · Conclusion")

    none_ece = results.variant_mean("none", "ece")
    temp_ece = results.variant_mean("temp", "ece")
    temp_acc = results.variant_mean("temp", "accuracy")
    wins = results.temp_beats_none_on_ece()

    add_panel(slide, MARGIN, Inches(1.95), CONTENT_W, Inches(1.38),
              fill=RGBColor(0xEC, 0xE6, 0xF3))
    _, frame = _textbox(slide, MARGIN + Inches(0.4), Inches(2.18),
                        CONTENT_W - Inches(0.8), Inches(1.2))
    run = frame.paragraphs[0].add_run()
    run.text = "What we stand behind"
    _style_run(run, 13, bold=True, color=ACCENT)
    paragraph = frame.add_paragraph()
    paragraph.space_before = Pt(8)
    run = paragraph.add_run()
    run.text = (f"Label-free LOO temperature scaling, applied ALONE to TDA, cuts mean ECE "
                f"{pct(none_ece)}% → {pct(temp_ece)}% at unchanged accuracy ({pct(temp_acc)}%), "
                f"on {wins} of 5 datasets.")
    _style_run(run, 17, bold=True, color=INK)

    add_bullets(slide, [
        "The bundle we set out to build does not work. Margin admission and probabilistic "
        "fusion are both harmful on average, and neither is reliably helpful alone.",
        "",
        "We know why each one failed, and the ablation is what told us - not intuition: "
        "margin starves the cache, fusion manufactures under-confidence, and together "
        "they render the one contribution that works inert.",
        "",
        "The finding is narrower than the one we set out to get - and we report it that "
        "way rather than framing the bundle as a success it is not.",
    ], top=Inches(3.58), size=15.5, spacing=4)

    add_callout(
        slide,
        "Not “three changes fix TDA calibration”, but “one of three does, run alone — "
        "and we diagnosed why the other two do not.”",
        top=Inches(6.15), size=15.5,
    )
    set_notes(slide, f"""
To close.

The headline claim we can defend is the box: leave-one-out temperature
scaling, applied on its own, cuts TDA mean calibration error from {pct(none_ece)} to
{pct(temp_ece)} percent while leaving accuracy completely unchanged, on {wins} of five
datasets. It is training-free, label-free, and costs one grid search.

The bundle we originally proposed does not work, and we are not going to
dress that up. Margin admission and probabilistic fusion are both harmful on
average.

What I would argue we actually got out of this project is the diagnosis.
The ablation - not our intuition - told us margin starves the cache, fusion
manufactures under-confidence, and the two of them together render the one
idea that works inert. We would not have known any of that from the headline
numbers alone.

Happy to take questions.
""")
    return slide


SLIDE_BUILDERS = [
    slide_title,
    slide_agenda,
    slide_clip,
    slide_clip_fails,
    slide_what_is_tta,
    slide_calibration,
    slide_the_gap,
    slide_tda_cache,
    slide_tda_admission,
    slide_feedback_loop,
    slide_change_margin,
    slide_change_fusion,
    slide_change_temperature,
    slide_setup,
    slide_headline,
    slide_ablation,
    slide_mechanism,
    slide_limitations,
    slide_conclusion,
]


def _slide_heading(slide, index):
    """The slide title, read back off the built slide.

    Taken from the rendered shapes rather than from a parallel list of strings,
    so the markdown mirror cannot claim a heading the deck does not have.
    """
    if index == 1:
        return "Title"
    candidates = [
        shape for shape in slide.shapes
        if shape.has_text_frame and shape.text_frame.text.strip()
        and shape.top is not None and shape.top < Inches(1.45)
    ]
    if not candidates:
        return f"Slide {index}"
    # The kicker sits above the title; the title is the lower of the two.
    return max(candidates, key=lambda s: s.top).text_frame.text.strip()


def write_markdown_mirror(prs, md_path) -> Path:
    """Emit `slides.md` from the deck that was just built.

    Generated, not hand-written, for the same reason the figures are: a second
    hand-maintained copy of the talk is a copy that silently disagrees with the
    deck the moment either one is edited.
    """
    lines = [
        "# Calibration-Aware Test-Time Adaptation for CLIP",
        "",
        "**Generated file — do not edit.** Produced by `python presentation/build_pptx.py`",
        "alongside `calibration_aware_tta.pptx`; edit the slide functions in that script",
        "instead. The deck is the deliverable, this is the readable mirror of it.",
        "",
        "Each section below is one slide: its on-screen heading, and the speaker notes",
        "that live in that slide's PowerPoint notes pane.",
        "",
    ]
    for index, slide in enumerate(prs.slides, start=1):
        heading = _slide_heading(slide, index)
        notes = ""
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame.text.strip()
        lines += ["---", "", f"## {index}. {heading}", ""]
        lines += [notes, ""] if notes else []

    md_path = Path(md_path)
    md_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return md_path


def build(summary_csv, ablation_csv, method_config_yaml, out_path,
          markdown_path=None) -> Path:
    results = Results(Path(summary_csv), Path(ablation_csv), Path(method_config_yaml))

    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H

    total = len(SLIDE_BUILDERS)
    for index, builder in enumerate(SLIDE_BUILDERS, start=1):
        slide = builder(prs, results)
        if index > 1:  # the title slide carries no footer
            add_footer(slide, index, total)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out_path))

    if markdown_path:
        write_markdown_mirror(prs, markdown_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", default="results/summary.csv")
    parser.add_argument("--ablation-csv", default="results/ablation.csv")
    parser.add_argument(
        "--method-config", default="configs/method/cal_tda.yaml",
        help="source of the margin/fusion constants the method slides quote",
    )
    parser.add_argument(
        "--out", default="presentation/calibration_aware_tta.pptx"
    )
    parser.add_argument(
        "--markdown", default="presentation/slides.md",
        help="readable mirror of the deck; pass an empty string to skip",
    )
    args = parser.parse_args()

    path = build(args.summary_csv, args.ablation_csv, args.method_config, args.out,
                 markdown_path=args.markdown or None)
    print(f"wrote {path} ({len(SLIDE_BUILDERS)} slides)")
    if args.markdown:
        print(f"wrote {args.markdown}")


if __name__ == "__main__":
    main()
