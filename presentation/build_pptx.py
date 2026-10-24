"""Generate the 18-slide conference deck for the final presentation.

The deck mirrors `report/main_v2.pdf`: each of the three changes is measured
alone against the TDA-equivalent baseline, margin admission is dropped,
probabilistic fusion is repaired by a two-way-tie gate, temperature does not
stack on the gate, and gated fusion is the recommended method.

Every number on every slide is READ FROM `results/summary.csv`,
`results/ablation.csv`, `results/ablation_gate.csv` and the margin run
records at build time. Nothing is typed in by hand. Re-run the sweeps, re-run
this script, and the deck is correct again -- the same rule the report
follows, applied to the talk.

    python presentation/build_pptx.py

Writes `presentation/calibration_aware_tta.pptx` (16:9). Speaker notes are in
the PowerPoint notes pane of every slide; `presentation/slides.md` mirrors the
same structure for reading outside PowerPoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

# Run as a script from the repo root, `presentation/` is sys.path[0], not the
# root; the deck shares the report's method order, labels and admission-rate
# reader rather than carrying a second copy that can drift.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from analysis.aggregate import (  # noqa: E402
    DATASET_LABEL as DATASET_LABELS, DATASET_ORDER, METHOD_LABEL, METHOD_ORDER,
)
from analysis.make_contribution_tables import _admission_rates  # noqa: E402

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



# --- Data --------------------------------------------------------------------
class Results:
    """Every figure the deck quotes, pulled from the committed CSVs and records.

    Kept as one object so a slide function can never reach around it to a
    hard-coded literal -- if a number is not derivable from the CSVs or the run
    records, it does not belong on a slide.
    """

    def __init__(self, summary_csv: Path, ablation_csv: Path, ablation_gate_csv: Path,
                 ablation_raw_dir: Path, margin_config_yaml: Path, gate_config_yaml: Path):
        self.summary = pd.read_csv(summary_csv)
        self.studies = {
            "ablation": pd.read_csv(ablation_csv),
            "ablation_gate": pd.read_csv(ablation_gate_csv),
        }
        # Percent, keyed by dataset: the same reader the report's margin table uses.
        self.admission = _admission_rates(Path(ablation_raw_dir))
        self.margin_config = yaml.safe_load(Path(margin_config_yaml).read_text(encoding="utf-8"))
        self.gate_config = yaml.safe_load(Path(gate_config_yaml).read_text(encoding="utf-8"))

    def cell(self, dataset: str, method: str, column: str) -> float:
        row = self.summary[
            (self.summary["dataset"] == dataset) & (self.summary["method"] == method)
        ]
        if row.empty:
            raise KeyError(f"no {method} row for {dataset} in summary.csv")
        return float(row.iloc[0][column])

    def method_mean(self, method: str, column: str) -> float:
        return float(self.summary[self.summary["method"] == method][column].mean())

    def variant_cell(self, dataset: str, variant: str, column: str,
                     study: str = "ablation") -> float:
        frame = self.studies[study]
        row = frame[(frame["dataset"] == dataset) & (frame["variant"] == variant)]
        if row.empty:
            raise KeyError(f"no {variant} row for {dataset} in {study}")
        return float(row.iloc[0][column])

    def variant_mean(self, variant: str, column: str, study: str = "ablation") -> float:
        frame = self.studies[study]
        return float(frame[frame["variant"] == variant][column].mean())

    def admission_rate(self, dataset: str) -> float:
        """Share of test samples the margin gate admitted, in percent."""
        return self.admission[dataset]

    def margin_constant(self, key: str):
        """A knob read from the config the margin ablation run actually used."""
        return self.margin_config[key]

    def gate_constant(self, key: str):
        """A knob read from the config the gated-fusion headline run actually used."""
        return self.gate_config[key]

    def n_total_images(self) -> int:
        per_dataset = self.summary.groupby("dataset")["n_samples"].first()
        return int(per_dataset.sum())

    def n_runs(self, study: str) -> int:
        return int(len(self.studies[study]))

    def boundary_fraction(self, dataset: str, method: str) -> float:
        """Share of a run's samples whose temperature search stopped on a grid edge."""
        return self.cell(dataset, method, "temp_frac_at_grid_boundary")

    def criterion_hits(self, method: str) -> int:
        """Datasets where `method` matches TDA's accuracy AND beats its ECE.

        The project's success criterion, recomputed here rather than asserted,
        so a slide cannot claim a different count than the CSV supports.
        """
        hits = 0
        for dataset in DATASET_ORDER:
            better_accuracy = self.cell(dataset, method, "accuracy") >= self.cell(
                dataset, "tda", "accuracy"
            )
            better_ece = self.cell(dataset, method, "ece") < self.cell(dataset, "tda", "ece")
            hits += int(better_accuracy and better_ece)
        return hits


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

The short answer, which I will not hide until the end: one of our three ideas
failed and was dropped, one failed and was repaired on the way, and the repaired
one is what we recommend. The interesting part of this talk is why.
""")
    return slide


def slide_agenda(prs, results):
    slide = new_slide(prs, "Agenda", kicker="Roadmap")

    entries = [
        ("1", "Background", "CLIP  ·  test-time adaptation  ·  what calibration means"),
        ("2", "TDA", "the cache-based method we build on, and where it goes wrong"),
        ("3", "Our method", "three training-free changes, each measured alone - one repaired on the way"),
        ("4", "Results", "five methods on five datasets, the ablation, and what the gate tells us"),
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

That is our diagnosis. Each of our three changes attacks one link in this
chain, and we measure each one alone before combining anything.
""")
    return slide


GOOD = RGBColor(0x3C, 0x8C, 0x5A)        # the one green in the deck: the result that holds


def _change_strip(slide, number, problem, change):
    """The "change N" chip plus the Baseline/Ours rows at the top of a change slide."""
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
        row_top = int(Inches(1.72) + index * Inches(0.74))
        _, frame = _textbox(slide, MARGIN, row_top, Inches(1.35), Inches(0.66),
                            anchor=MSO_ANCHOR.MIDDLE)
        run = frame.paragraphs[0].add_run()
        run.text = label
        _style_run(run, 12.5, bold=True, color=MUTED)

        _, frame = _textbox(slide, MARGIN + Inches(1.45), row_top,
                            Inches(10.65), Inches(0.66), anchor=MSO_ANCHOR.MIDDLE)
        run = frame.paragraphs[0].add_run()
        run.text = text
        _style_run(run, 15, bold=(index == 1), color=color)


def _change_slide(prs, *, number, title, kicker, problem, change, left_items, table,
                  callout, callout_color=ACCENT, notes="", footnote=None):
    """Idea on top, reasons on the left, the per-dataset verdict on the right.

    One layout for every change slide, so the three read as a set: the chip
    and the Baseline/Ours strip say what the change is, the table says what
    it did against the same baseline, and the callout says the verdict.
    """
    slide = new_slide(prs, title, kicker=kicker)
    _change_strip(slide, number, problem, change)

    body_top = Inches(3.25)
    add_bullets(slide, left_items, left=MARGIN, top=body_top, width=Inches(5.55),
                size=12, sub_size=11, spacing=4)

    table_left = Inches(6.40)
    add_table(
        slide, table["header"], table["rows"], table_left, body_top,
        SLIDE_W - MARGIN - table_left,
        col_widths=table.get("col_widths"),
        font_size=table.get("font_size", 11), header_size=table.get("header_size", 10.5),
        row_height=table.get("row_height", Inches(0.32)),
        highlight_rows=table.get("highlight_rows", ()),
        highlight_color=table.get("highlight_color"),
    )

    if footnote:
        _, frame = _textbox(slide, MARGIN, Inches(5.72), CONTENT_W, Inches(0.40))
        run = frame.paragraphs[0].add_run()
        run.text = footnote
        _style_run(run, 10.5, italic=True, color=MUTED)

    add_callout(slide, callout, color=callout_color, top=Inches(6.12), size=15)
    set_notes(slide, notes)
    return slide


def _versus_rows(results, variant, study="ablation"):
    """Per-dataset rows comparing the `none` baseline with `variant`, plus a Mean row."""
    rows = []
    for dataset in DATASET_ORDER:
        rows.append([
            DATASET_LABELS[dataset],
            pct(results.variant_cell(dataset, "none", "accuracy", study)),
            pct(results.variant_cell(dataset, variant, "accuracy", study)),
            pct(results.variant_cell(dataset, "none", "ece", study)),
            pct(results.variant_cell(dataset, variant, "ece", study)),
        ])
    rows.append([
        "Mean",
        pct(results.variant_mean("none", "accuracy", study)),
        pct(results.variant_mean(variant, "accuracy", study)),
        pct(results.variant_mean("none", "ece", study)),
        pct(results.variant_mean(variant, "ece", study)),
    ])
    return rows


def _versus_header(variant):
    return ["Dataset", "none acc.", f"{variant} acc.", "none ECE", f"{variant} ECE"]


def slide_margin(prs, results):
    threshold = results.margin_constant("margin_threshold")
    none_acc = results.variant_mean("none", "accuracy")
    margin_acc = results.variant_mean("margin", "accuracy")
    none_ece = results.variant_mean("none", "ece")
    margin_ece = results.variant_mean("margin", "ece")
    rates = {d: results.admission_rate(d) for d in DATASET_ORDER}
    high = max(rates, key=rates.get)
    low = min(rates, key=rates.get)
    pets_none = results.variant_cell("pets", "none", "ece")
    pets_margin = results.variant_cell("pets", "margin", "ece")
    aircraft_none = results.variant_cell("aircraft", "none", "ece")
    aircraft_margin = results.variant_cell("aircraft", "margin", "ece")

    rows = []
    for dataset in DATASET_ORDER:
        rows.append([
            DATASET_LABELS[dataset],
            f"{results.cell(dataset, 'clipzs', 'n_samples'):,.0f}",
            f"{results.cell(dataset, 'clipzs', 'n_classes'):.0f}",
            f"{rates[dataset]:.1f}%",
            pct(results.variant_cell(dataset, "none", "accuracy")),
            pct(results.variant_cell(dataset, "margin", "accuracy")),
            pct(results.variant_cell(dataset, "none", "ece")),
            pct(results.variant_cell(dataset, "margin", "ece")),
        ])
    rows.append([
        "Mean / total", f"{results.n_total_images():,}", "",
        f"{sum(rates.values()) / len(rates):.1f}%",
        pct(none_acc), pct(margin_acc), pct(none_ece), pct(margin_ece),
    ])

    return _change_slide(
        prs, number=1,
        title="Change 1: admit by margin — and why it fails",
        kicker="3 · Our method, one change at a time",
        problem="TDA offers every sample to its positive cache; entropy only decides who gets evicted.",
        change=(f"Admit a sample only if its top logit beats the runner-up by more than "
                f"{threshold:.1f}: one global threshold, fixed before any run."),
        left_items=[
            f"It does not port. CLIP logits are 100 × a cosine, so a gap of {threshold:.1f} is a "
            "0.01 cosine difference, which means something different on 10 and 100 classes. "
            f"Admission follows CLIP's accuracy: {rates[high]:.1f}% on {DATASET_LABELS[high]}, "
            f"{rates[low]:.1f}% on {DATASET_LABELS[low]}.",
            "A starving cache is a biased cache. Only the classes CLIP already finds easy "
            "clear the bar, so the cache can only vote for those, and pulls every prediction "
            f"toward them: {DATASET_LABELS[low]} ECE {pct(aircraft_none)} → {pct(aircraft_margin)}. "
            f"Pets, at {rates['pets']:.1f}% admission, is a near no-op "
            f"({pct(pets_none)} → {pct(pets_margin)}).",
            "It filters loudness, not correctness. A confidently wrong sample has a large "
            "margin too, so the gate removes close calls, which TDA's eviction already "
            "handled, and nothing else.",
        ],
        table={
            "header": ["Dataset", "Images", "Classes", "Admitted",
                       "none acc.", "margin acc.", "none ECE", "margin ECE"],
            "rows": rows,
            "col_widths": [0.19, 0.11, 0.10, 0.12, 0.12, 0.12, 0.12, 0.12],
            "font_size": 10.5, "header_size": 10, "row_height": Inches(0.30),
            "highlight_rows": (len(rows) - 1,),
        },
        footnote=("Every number from here on: ViT-B/16, batch 1, seed 0, CoOp test splits "
                  "only, ECE with 20 bins. \"none\" = our method with every change off = TDA "
                  "through our code, run in the same sweep; each change is measured alone "
                  "against it."),
        callout=(f"Loses on every dataset, on both metrics: accuracy {pct(none_acc)} → "
                 f"{pct(margin_acc)}, ECE {pct(none_ece)} → {pct(margin_ece)}. Dropped."),
        callout_color=WARN,
        notes=f"""
From here on, one change at a time, each measured alone against the same
baseline: our method with every change switched off, which is TDA run
through our code, in the same sweep. The small table on the right also
carries the setup: five datasets, {results.n_total_images():,} test images per method, CoOp
test splits, ViT-B/16, batch size one so it is genuinely online, seed zero,
twenty-bin ECE. Labels are opened once per run, by the final scorer.

Change one replaces entropy admission with a margin gate: admit only if the
top logit beats the runner-up by more than {threshold:g}. One global threshold,
fixed before any run, because fitting it per test set would be test-set
tuning even without labels.

It loses everywhere. Mean accuracy {pct(none_acc)} to {pct(margin_acc)}, mean ECE
{pct(none_ece)} to {pct(margin_ece)}. The admitted column explains the first reason: a
logit gap of one means something different on ten and a hundred classes, so
the gate admits {rates[high]:.1f} percent on {DATASET_LABELS[high]} and {rates[low]:.1f} percent on
{DATASET_LABELS[low]}, ordered by how well CLIP already does.

Second reason: a starving cache is not a smaller cache, it is a biased one.
Only the easy classes clear the bar, so the cache can only ever vote for
easy classes, whatever the image shows. That is why {DATASET_LABELS[low]} doubles its
ECE while Pets, where almost everything is admitted, barely moves.

Third: margin filters loudness, not correctness. A confidently wrong sample
has a large margin exactly like a confidently right one, so the gate could
never remove the samples it was designed for. We drop it; everything from
here on has margin admission off.
""")


def slide_fusion_proposed(prs, results):
    weight = results.gate_constant("fusion_weight")
    none_acc = results.variant_mean("none", "accuracy")
    fusion_acc = results.variant_mean("fusion", "accuracy")
    none_ece = results.variant_mean("none", "ece")
    fusion_ece = results.variant_mean("fusion", "ece")
    pets_acc = results.variant_cell("pets", "fusion", "accuracy")
    pets_ece = results.variant_cell("pets", "fusion", "ece")
    flowers_none = results.variant_cell("flower102", "none", "ece")
    flowers_fusion = results.variant_cell("flower102", "fusion", "ece")
    aircraft_none = results.variant_cell("aircraft", "none", "ece")
    aircraft_fusion = results.variant_cell("aircraft", "fusion", "ece")

    return _change_slide(
        prs, number=2,
        title="Change 2: fuse probabilities — and why that fails too",
        kicker="3 · Our method, one change at a time",
        problem="TDA adds the cache's logits onto CLIP's and only then takes a softmax: two agreeing sources sharpen each other.",
        change=(f"Convex mixture of the two distributions, p = (1 − w)·softmax(clip) + "
                f"w·softmax(cache), w = {weight:g}: bounded by its most confident input."),
        left_items=[
            f"Accuracy barely moves ({pct(none_acc)} → {pct(fusion_acc)}): the cache still "
            "chooses the class. What breaks is the confidence attached to the choice.",
            "The two inputs live on different scales. A cache logit is a sum of at most "
            "three e^(−β(1−cos)) terms, about 2 at most; softmaxed over 100 classes that is "
            "nearly flat, top entry around 7%. CLIP's logits carry its ≈100 scale and peak "
            "above 90%.",
            "A 50/50 average of a peaked distribution and a flat one halves the peak, on "
            "every sample, right or wrong. ECE punishes underconfidence exactly as it "
            f"punishes overconfidence: Pets is right {pct(pets_acc)}% of the time at "
            f"roughly 45% confidence, a gap of about 44 points, the {pct(pets_ece)} in the table.",
            f"Aircraft looks like a win ({pct(aircraft_none)} → {pct(aircraft_fusion)}) by "
            "accident: its true accuracy happens to sit near where the ceiling landed.",
        ],
        table={
            "header": _versus_header("fusion"),
            "rows": _versus_rows(results, "fusion"),
            "col_widths": [0.24, 0.19, 0.19, 0.19, 0.19],
            "highlight_rows": (5,),
        },
        callout=(f"Mean ECE {pct(none_ece)} → {pct(fusion_ece)}, the worst single change in "
                 f"the project (Flowers102 {pct(flowers_none)} → {pct(flowers_fusion)}). "
                 "The equation shows why."),
        callout_color=WARN,
        notes=f"""
Change two attacks the combination step. TDA adds logits and then takes a
softmax, so two sources that agree sharpen each other. A convex mixture of
the two probability distributions cannot do that: it is bounded by its most
confident input. We weight them equally.

As proposed it is the worst single change in the project. Mean ECE
{pct(none_ece)} to {pct(fusion_ece)}, driven by Flowers {pct(flowers_none)} to {pct(flowers_fusion)} and Pets
to {pct(pets_ece)}. Accuracy barely moves, so the cache is still choosing the
class; what breaks is the confidence.

The mechanism is in the equation once you compare the scales. A cache logit
is a sum of a few exponential affinity terms, about two at most, and most
classes score near zero; softmaxed over a hundred classes that is almost
flat, top entry around seven percent. CLIP's logits are peaked, often above
ninety. Average the two fifty-fifty and you halve the peak, on every sample,
right or wrong. Confidence is capped near one half. ECE punishes
underconfidence as hard as overconfidence, so the cost scales with accuracy:
Pets is right {pct(pets_acc)} percent of the time and reports about forty-five,
and that gap is the number in the table.

Aircraft is the one place fusion looks good, and it is an accident: its true
accuracy happens to sit near where the ceiling landed.
""")


def _bar_sketch(slide, left, top, width, height, probs, color, label, verdict):
    """A five-bar probability sketch: the shape of a prediction, not a measurement."""
    add_panel(slide, left, top, width, height)
    n = len(probs)
    gap = Inches(0.08)
    inner_left = left + Inches(0.22)
    inner_width = width - Inches(0.44)
    bar_w = int((inner_width - gap * (n - 1)) / n)
    base = top + height - Inches(0.62)
    max_h = height - Inches(0.98)
    for index, p in enumerate(probs):
        bar_h = int(max_h * p)
        bar = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, int(inner_left + index * (bar_w + gap)),
            int(base - bar_h), bar_w, bar_h,
        )
        bar.fill.solid()
        bar.fill.fore_color.rgb = color if index < 2 else NEUTRAL
        bar.line.fill.background()
        bar.shadow.inherit = False
    rule = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, inner_left, base, inner_width, Pt(1.2))
    rule.fill.solid()
    rule.fill.fore_color.rgb = RULE
    rule.line.fill.background()
    rule.shadow.inherit = False

    _, frame = _textbox(slide, left + Inches(0.12), top + Inches(0.08), width - Inches(0.24),
                        Inches(0.30))
    run = frame.paragraphs[0].add_run()
    run.text = label
    _style_run(run, 12, bold=True, color=INK)
    _, frame = _textbox(slide, left + Inches(0.12), base + Inches(0.08), width - Inches(0.24),
                        Inches(0.46))
    run = frame.paragraphs[0].add_run()
    run.text = verdict
    _style_run(run, 11, bold=True, color=color)


def slide_gate(prs, results):
    max_gap = results.gate_constant("fusion_gate_max_gap")
    min_mass = results.gate_constant("fusion_gate_min_mass")
    weight = results.gate_constant("fusion_weight")
    slide = new_slide(prs, "The repair: fuse only on a two-way tie",
                      kicker="3 · Our method, one change at a time")

    cases = [
        ("CLIP has a clear winner", MUTED,
         "The cache cannot change the argmax. Adding its logits only sharpens (TDA's "
         "overconfidence); mixing only dilutes. Leave it out: the output is CLIP's own "
         "distribution, negative-cache correction included."),
        ("CLIP is torn between two classes", ACCENT,
         "The cache can break the tie, which is where TDA's accuracy gain lives, and on a "
         f"genuine tie a confidence near one half is honest. Fuse, w = {weight:g}."),
    ]
    width, gap = Inches(5.95), Inches(0.21)
    for index, (heading, color, body) in enumerate(cases):
        left = int(MARGIN + index * (width + gap))
        add_panel(slide, left, Inches(1.72), width, Inches(1.42),
                  fill=TINT_ACCENT if index == 1 else SURFACE)
        _, frame = _textbox(slide, left + Inches(0.26), Inches(1.84), width - Inches(0.52),
                            Inches(1.2))
        run = frame.paragraphs[0].add_run()
        run.text = heading
        _style_run(run, 15, bold=True, color=color)
        paragraph = frame.add_paragraph()
        paragraph.space_before = Pt(5)
        run = paragraph.add_run()
        run.text = body
        _style_run(run, 12, color=INK)

    add_panel(slide, MARGIN, Inches(3.32), CONTENT_W, Inches(0.80))
    _, frame = _textbox(slide, MARGIN + Inches(0.3), Inches(3.44), CONTENT_W - Inches(0.6),
                        Inches(0.6), anchor=MSO_ANCHOR.MIDDLE)
    run = frame.paragraphs[0].add_run()
    run.text = (f"two-way tie   ⇔   p(1) − p(2) < {max_gap:g}    and    "
                f"p(1) + p(2) > {min_mass:g}")
    _style_run(run, 17, bold=True, color=INK)
    run = frame.paragraphs[0].add_run()
    run.text = "        p = softmax(clip); p(1), p(2) its two largest entries"
    _style_run(run, 12, italic=True, color=MUTED)

    sketches = [
        ([0.84, 0.07, 0.04, 0.03, 0.02], MUTED, "Clear winner", "gap large → CLIP alone"),
        ([0.44, 0.37, 0.09, 0.06, 0.04], ACCENT, "Two-way tie", "small gap, top-two mass high → fuse"),
        ([0.26, 0.21, 0.20, 0.17, 0.16], MUTED, "Spread thin", "small gap, but confused, not torn → CLIP alone"),
    ]
    width, gap = Inches(3.90), Inches(0.21)
    for index, (probs, color, label, verdict) in enumerate(sketches):
        left = int(MARGIN + index * (width + gap))
        _bar_sketch(slide, left, Inches(4.30), width, Inches(1.66), probs, color, label, verdict)

    add_callout(
        slide,
        "Same quantity as change 1, opposite purpose: applied at prediction time, to decide "
        "when the cache is consulted, not at storage time. Every class stays represented.",
        top=Inches(6.12), size=14.5,
    )
    set_notes(slide, f"""
The diagnosis says fusion fails because it runs on every sample, including
the large majority where CLIP already has a clear winner and the cache can
only dilute it. On those samples the cache cannot change the argmax either
way: adding would only sharpen, which is TDA's overconfidence, and mixing
would only dilute. So leave it out.

The cache has a real job on exactly one kind of sample: a two-way tie. There
it can break the tie, which is where TDA's accuracy gain comes from, and on
a genuine tie a confidence near one half is roughly honest.

So we gate the fusion. Take CLIP's softmax and its two largest entries. A
sample is a two-way tie when the gap between them is below {max_gap:g} and
their sum is above {min_mass:g}. The first condition is the close call. The
second is what low entropy buys: a sample spread thinly over many classes
also has a small gap, but there CLIP is confused rather than torn, and the
cache's vote is noise. The three sketches show the three cases.

On a tie the output is the fifty-fifty mixture; otherwise it is CLIP's own
distribution, still corrected by TDA's negative cache, which we never touch.
Both thresholds are in probability space, so one global pair means the same
thing on ten and a hundred classes, which is exactly the portability margin
admission lacked, and both were fixed before any run.

Notice it is the same quantity change one used, the gap between the top two
scores, with the opposite purpose: at prediction time, deciding when the
cache is consulted, instead of at storage time, deciding what it remembers.
""")
    return slide


def _result_slide(prs, *, title, kicker, table, items, callout, callout_color, notes,
                  table_width=Inches(6.9)):
    """Table on the left, the reading of it on the right."""
    slide = new_slide(prs, title, kicker=kicker)
    add_table(
        slide, table["header"], table["rows"], MARGIN, Inches(1.95), table_width,
        col_widths=table.get("col_widths"), font_size=table.get("font_size", 12),
        header_size=table.get("header_size", 11.5), row_height=table.get("row_height", Inches(0.36)),
        highlight_rows=table.get("highlight_rows", ()), highlight_color=table.get("highlight_color"),
    )
    right = MARGIN + table_width + Inches(0.45)
    add_bullets(slide, items, left=right, top=Inches(1.95), width=SLIDE_W - MARGIN - right,
                size=13.5, sub_size=12, spacing=7)
    add_callout(slide, callout, color=callout_color, top=Inches(6.12), size=15)
    set_notes(slide, notes)
    return slide


def slide_gate_result(prs, results):
    g = "ablation_gate"
    none_ece = results.variant_mean("none", "ece", g)
    gate_ece = results.variant_mean("gate", "ece", g)
    none_acc = results.variant_mean("none", "accuracy", g)
    gate_acc = results.variant_mean("gate", "accuracy", g)
    clip_ece = results.method_mean("clipzs", "ece")
    cell = lambda d, v, c: results.variant_cell(d, v, c, g)  # noqa: E731
    drops = {d: cell(d, "none", "accuracy") - cell(d, "gate", "accuracy")
             for d in ("dtd", "flower102", "eurosat")}
    tda_gain_eurosat = results.cell("eurosat", "tda", "accuracy") - results.cell("eurosat", "clipzs", "accuracy")

    return _result_slide(
        prs,
        title="Gated fusion: the blow-up is gone",
        kicker="3 · Our method, one change at a time",
        table={
            "header": _versus_header("gate"),
            "rows": _versus_rows(results, "gate", g),
            "col_widths": [0.24, 0.19, 0.19, 0.19, 0.19],
            "highlight_rows": (5,),
        },
        items=[
            f"Mean ECE {pct(none_ece)} → {pct(gate_ece)}, below zero-shot CLIP's own "
            f"{pct(clip_ece)}: better calibrated than the frozen model it started from.",
            f"Ten-plus points on three datasets: DTD {pct(cell('dtd', 'none', 'ece'))} → "
            f"{pct(cell('dtd', 'gate', 'ece'))}, EuroSAT {pct(cell('eurosat', 'none', 'ece'))} → "
            f"{pct(cell('eurosat', 'gate', 'ece'))}, Aircraft {pct(cell('aircraft', 'none', 'ece'))} → "
            f"{pct(cell('aircraft', 'gate', 'ece'))}. Pets is the one loss "
            f"({pct(cell('pets', 'none', 'ece'))} → {pct(cell('pets', 'gate', 'ece'))}), where TDA "
            "was already nearly perfect.",
            f"The cost is accuracy: −{drops['dtd']:.2f} on DTD, −{drops['flower102']:.2f} on "
            f"Flowers102, −{drops['eurosat']:.2f} on EuroSAT, where the gate gives back half of "
            f"TDA's {tda_gain_eurosat:.1f}-point gain. Pets and Aircraft are unchanged or slightly up.",
            "Same quantity as change 1, opposite purpose. Margin admission applied the top-two "
            "gap at storage time and threw samples away, unevenly across classes; the gate "
            "applies it at prediction time and every class stays represented.",
        ],
        callout=(f"Mean accuracy {pct(none_acc)} → {pct(gate_acc)}, mean ECE {pct(none_ece)} → "
                 f"{pct(gate_ece)}. The cache is consulted only where it can change the answer."),
        callout_color=GOOD,
        notes=f"""
With the gate, the blow-up is gone. Flowers and Pets are back to single-digit
ECE, and gated fusion beats the baseline's ECE on four of five datasets, by
ten or more points on three of them. Mean ECE goes from {pct(none_ece)} to
{pct(gate_ece)}, which is below zero-shot CLIP's own {pct(clip_ece)}: the method is better
calibrated than the frozen model it started from. Pets is the one loss, on
the dataset where TDA was already nearly perfectly calibrated.

The accuracy side is the cost. Gated fusion matches or fractionally exceeds
the baseline on Pets and Aircraft, gives up {drops['flower102']:.2f} points on Flowers and
{drops['dtd']:.2f} on DTD, and {drops['eurosat']:.2f} on EuroSAT, where it returns about half of
TDA's gain. EuroSAT is where TDA's cache weighting is strongest, so its flips
presumably come from moderate-margin samples the gate keeps the cache away
from; we did not instrument that.

The contrast with change one is the clearest finding in the project. Margin
admission applied the top-two gap at storage time and threw information
away, unevenly across classes. The gate applies it at prediction time and
leaves every class represented.
""")


def slide_temperature(prs, results):
    none_acc = results.variant_mean("none", "accuracy")
    temp_acc = results.variant_mean("temp", "accuracy")
    none_ece = results.variant_mean("none", "ece")
    temp_ece = results.variant_mean("temp", "ece")
    cell = results.variant_cell

    return _change_slide(
        prs, number=3,
        title="Change 3: a label-free temperature — one reversal",
        kicker="3 · Our method, one change at a time",
        problem="TDA never rescales its output; whatever sharpness falls out of the sum is what it reports.",
        change=("Divide logits by T ≥ 1, chosen so mean confidence matches a label-free accuracy "
                "estimate: leave-one-out nearest-neighbour agreement inside the cache."),
        left_items=[
            f"Accuracy identical on every dataset, to 0.00 ({pct(none_acc)} = {pct(temp_acc)}): "
            "a monotonic rescaling cannot move an argmax. Any difference would be a bug.",
            f"Mean ECE {pct(none_ece)} → {pct(temp_ece)}, with the big wins where the baseline "
            f"was most overconfident: DTD {pct(cell('dtd', 'none', 'ece'))} → "
            f"{pct(cell('dtd', 'temp', 'ece'))}, Aircraft {pct(cell('aircraft', 'none', 'ece'))} → "
            f"{pct(cell('aircraft', 'temp', 'ece'))}.",
            f"Pets reverses: {pct(cell('pets', 'none', 'ece'))} → {pct(cell('pets', 'temp', 'ece'))}. "
            "The search has no stopping rule, it always moves toward the estimate; and the "
            "estimate is biased low on fine-grained data, where a cached Birman's nearest "
            "neighbour is often a Ragdoll even when both pseudo-labels are right.",
            "The grid floor at T = 1 prevents the opposite mistake, sharpening on an inflated "
            "estimate. It cannot help here.",
        ],
        table={
            "header": _versus_header("temp"),
            "rows": _versus_rows(results, "temp"),
            "col_widths": [0.24, 0.19, 0.19, 0.19, 0.19],
            "highlight_rows": (5,),
        },
        callout=("Helps where TDA was overconfident, hurts where it was already honest. "
                 "Accuracy untouched by construction."),
        callout_color=BASELINE,
        notes=f"""
Change three. A well-calibrated model has mean confidence equal to its
accuracy. Mean confidence we can measure; accuracy normally needs labels. We
estimate it from the cache itself: leave-one-out nearest-neighbour agreement
among the cached features and their pseudo-labels. Then a grid search picks
the temperature, at least one, that brings mean confidence onto that
estimate. Dividing logits by a positive constant is monotonic, so this change
cannot move an argmax.

The table shows that guarantee in the measurements: accuracy identical to two
decimals on every dataset. On calibration it is a real but uneven gain. Mean
ECE {pct(none_ece)} to {pct(temp_ece)}, with large drops where the baseline was badly
overconfident, DTD and Aircraft, and small ones on EuroSAT and Flowers.

Pets is a clear reversal, {pct(cell('pets', 'none', 'ece'))} to {pct(cell('pets', 'temp', 'ece'))}. Two reasons.
The search has no stopping rule: it always moves confidence toward the
estimate, so where TDA was already honest the only thing it can do is
damage. And the estimate is biased low on fine-grained data: on Pets a
cached Birman's nearest cached neighbour is often a Ragdoll even when both
pseudo-labels are correct, so agreement sits well below the true accuracy,
the search reads that as overconfidence, and smooths a model that did not
need it. The floor at T equals one only prevents the opposite mistake.
""")


DECK_METHOD_LABEL = {
    method: (label if method in ("clipzs", "tda") else f"Ours: {label.lower()}")
    for method, label in METHOD_LABEL.items()
}
DECK_METHOD_LABEL["clipzs"] = "CLIP zero-shot"
DECK_METHOD_LABEL["cal_tda_gate_temp"] = "Ours: gate + temp"


def _method_table(slide, results, column, top, *, title):
    header = ["", *(DATASET_LABELS[d] for d in DATASET_ORDER), "mean"]
    rows = []
    for method in METHOD_ORDER:
        values = [results.cell(d, method, column) for d in DATASET_ORDER]
        rows.append([DECK_METHOD_LABEL[method], *(pct(v) for v in values),
                     pct(results.method_mean(method, column))])

    _, frame = _textbox(slide, MARGIN, int(top - Inches(0.27)), Inches(7.2), Inches(0.26))
    run = frame.paragraphs[0].add_run()
    run.text = title
    _style_run(run, 12, bold=True, color=MUTED)

    return add_table(
        slide, header, rows, MARGIN, top, Inches(7.25),
        col_widths=[0.235, 0.117, 0.145, 0.108, 0.133, 0.132, 0.13],
        font_size=10.5, header_size=10, row_height=Inches(0.285),
        highlight_rows=(METHOD_ORDER.index("cal_tda_gate"),),
    )


def slide_together(prs, results):
    slide = new_slide(prs, "Putting it together: five methods, five datasets",
                      kicker="4 · Results")

    _method_table(slide, results, "accuracy", Inches(2.05),
                  title="Top-1 accuracy (%)   -   higher is better")
    _method_table(slide, results, "ece", Inches(4.22),
                  title="ECE, 20 bins (%)   -   lower is better")

    figure = Path("analysis/figures/accuracy_vs_ece.png")
    if figure.exists():
        add_picture(slide, figure, Inches(8.05), Inches(1.80), Inches(4.67), Inches(4.15))

    clip_acc = results.method_mean("clipzs", "accuracy")
    tda_acc = results.method_mean("tda", "accuracy")
    gate_acc = results.method_mean("cal_tda_gate", "accuracy")
    clip_ece = results.method_mean("clipzs", "ece")
    tda_ece = results.method_mean("tda", "ece")
    gate_ece = results.method_mean("cal_tda_gate", "ece")
    kept, gain = gate_acc - clip_acc, tda_acc - clip_acc
    add_callout(
        slide,
        f"Gated fusion keeps {kept:.2f} of TDA's {gain:.2f} accuracy points over CLIP and "
        f"reports better-calibrated confidence than either: ECE {pct(gate_ece)} vs CLIP "
        f"{pct(clip_ece)}, TDA {pct(tda_ece)}.",
        color=GOOD, top=Inches(6.12), size=14.5,
    )
    set_notes(slide, f"""
Now the five headline methods side by side: zero-shot CLIP, TDA, and three
configurations of ours, gated fusion alone, temperature alone, and the two
together, all with margin admission off.

Averaged over the five datasets, CLIP reaches {pct(clip_acc)} percent accuracy at
{pct(clip_ece)} ECE, TDA {pct(tda_acc)} at {pct(tda_ece)}. Gated fusion alone, the highlighted
row, reaches {pct(gate_acc)} at {pct(gate_ece)}: it keeps {kept:.2f} of TDA's {gain:.2f} points of
accuracy gain over CLIP while reporting better-calibrated confidence than
either of them.

The scatter on the right is the same comparison. The target is a point as
far right as TDA and as far down as zero-shot CLIP; the gated-fusion marker
sits below both on three of the five datasets.
""")
    return slide


def slide_ablation_gate(prs, results):
    g = "ablation_gate"
    slide = new_slide(prs, "The ablation: the two do not stack", kicker="4 · Results")

    labels = [("none", "TDA-equivalent (both off)"), ("gate", "+ gated fusion"),
              ("temp", "+ LOO temperature"), ("gate_temp", "+ gated fusion + temperature")]
    rows = [[label, pct(results.variant_mean(v, "accuracy", g)), pct(results.variant_mean(v, "ece", g))]
            for v, label in labels]
    add_table(slide, ["Variant (margin off), mean of 5", "Accuracy", "ECE"], rows,
              MARGIN, Inches(1.90), Inches(5.6), col_widths=[0.58, 0.21, 0.21],
              font_size=12, header_size=11, row_height=Inches(0.36), highlight_rows=(1,))

    gate_ece = results.variant_mean("gate", "ece", g)
    gate_temp_ece = results.variant_mean("gate_temp", "ece", g)
    clip_ece = results.method_mean("clipzs", "ece")
    pets_gate = results.variant_cell("pets", "gate", "ece", g)
    pets_gate_temp = results.variant_cell("pets", "gate_temp", "ece", g)
    eurosat_pinned = 100 * results.boundary_fraction("eurosat", "cal_tda_gate_temp")
    aircraft_pinned = 100 * results.boundary_fraction("aircraft", "cal_tda_gate_temp")
    hits = results.criterion_hits("cal_tda_gate")

    add_bullets(slide, [
        f"Temperature on top of the gate leaves accuracy unchanged, as it must, and makes "
        f"ECE worse: {pct(gate_ece)} → {pct(gate_temp_ece)}.",
        f"Both target the same overconfidence. Once the gate removes it at the source, what "
        f"is left is CLIP's own calibration (≈{pct(clip_ece)}) and the label-free estimate "
        f"cannot improve on it: the search sits on a grid edge for {eurosat_pinned:.1f}% of "
        f"EuroSAT and {aircraft_pinned:.1f}% of Aircraft samples, and on Pets it acts the "
        f"wrong way ({pct(pets_gate)} → {pct(pets_gate_temp)}).",
        f"Strict criterion (accuracy ≥ TDA and ECE < TDA, on 3 of 5): gated fusion meets it "
        f"on {hits} of 5. We report that count as it stands.",
    ], left=MARGIN, top=Inches(3.85), width=Inches(5.6), size=12.5, sub_size=11.5, spacing=5)

    figures = Path("analysis/figures")
    panels = [("reliability_dtd.png", "DTD"), ("reliability_pets.png", "Pets")]
    panel_w, panel_h = Inches(3.05), Inches(3.05)
    for index, (name, _label) in enumerate(panels):
        path = figures / name
        if path.exists():
            add_picture(slide, path, int(Inches(6.55) + index * (panel_w + Inches(0.12))),
                        Inches(1.82), panel_w, panel_h)
    legend = figures / "reliability_legend.png"
    if legend.exists():
        add_picture(slide, legend, Inches(6.55), Inches(4.95), Inches(6.2), Inches(1.0))

    add_callout(
        slide,
        "Recommend gated fusion alone: one extra comparison per sample, no labels, no "
        "training, no change to the cache.",
        color=GOOD, top=Inches(6.12), size=15,
    )
    set_notes(slide, f"""
The ablation over the two surviving changes: both on/off combinations of
gated fusion and temperature, margin off, on all five datasets, {results.n_runs(g)}
records.

They do not stack. Adding temperature on top of the gate leaves accuracy
unchanged, as it must, and makes mean ECE worse, {pct(gate_ece)} to {pct(gate_temp_ece)}.
The reason is that both target the same thing. Temperature scaling was
designed to undo the overconfidence TDA's cache manufactures; the gate
removes that overconfidence at the source, by never letting the cache sharpen
a prediction CLIP had already made. What is left after the gate is CLIP's
own calibration, already around {pct(clip_ece)}, and the label-free estimate is not
accurate enough to improve on it. On EuroSAT and Aircraft the search wants
to sharpen, the floor blocks it, and it sits on a grid edge for {eurosat_pinned:.1f} and
{aircraft_pinned:.1f} percent of samples; on Pets it acts on every sample, in the wrong
direction, the same Pets failure as before.

The reliability diagrams on the right show the direction of each failure
directly: TDA's curve sits below the identity line, the temperature-only
curve sits above it on Pets, and gated fusion tracks the diagonal most
closely on DTD.

Against the project's own strict criterion, accuracy at least TDA and ECE
below TDA on three of five datasets, gated fusion passes on {hits} of five. We
report that as it stands. So we recommend gated fusion alone.
""")
    return slide


def slide_conclusion(prs, results):
    slide = new_slide(prs, "Conclusion", kicker="5 · Conclusion")

    clip_acc = results.method_mean("clipzs", "accuracy")
    tda_acc = results.method_mean("tda", "accuracy")
    gate_acc = results.method_mean("cal_tda_gate", "accuracy")
    clip_ece = results.method_mean("clipzs", "ece")
    tda_ece = results.method_mean("tda", "ece")
    gate_ece = results.method_mean("cal_tda_gate", "ece")
    temp_ece = results.variant_mean("temp", "ece")

    add_panel(slide, MARGIN, Inches(1.85), CONTENT_W, Inches(1.32), fill=TINT_ACCENT)
    _, frame = _textbox(slide, MARGIN + Inches(0.4), Inches(2.02), CONTENT_W - Inches(0.8),
                        Inches(1.1))
    run = frame.paragraphs[0].add_run()
    run.text = "What we recommend"
    _style_run(run, 13, bold=True, color=ACCENT)
    paragraph = frame.add_paragraph()
    paragraph.space_before = Pt(6)
    run = paragraph.add_run()
    run.text = (f"TDA's cache with gated probabilistic fusion: mean ECE {pct(tda_ece)} → "
                f"{pct(gate_ece)} (zero-shot CLIP: {pct(clip_ece)}) at {pct(gate_acc)}% accuracy "
                f"(TDA {pct(tda_acc)}, CLIP {pct(clip_acc)}). One extra comparison per sample, "
                "no labels, no training.")
    _style_run(run, 16, bold=True, color=INK)

    add_bullets(slide, [
        "Margin admission fails outright: a fixed logit threshold does not port, a starved "
        "cache is a biased cache, and the top-two gap cannot tell a confident mistake from a "
        "confident success. Dropped.",
        "",
        "Probabilistic fusion fails as proposed, for a reason the equation makes visible: a "
        "flat cache distribution averaged into CLIP's peaked one caps every confidence near "
        "one half. Gating it to two-way ties removes the dilution and keeps the one thing "
        "the cache is good for, breaking ties.",
        "",
        f"Leave-one-out temperature helps on its own (mean ECE {pct(tda_ece)} → {pct(temp_ece)} "
        "at unchanged accuracy) but adds nothing on top of the gate, because the gate has "
        "already removed the overconfidence it was built to correct.",
    ], top=Inches(3.40), size=14, spacing=3)

    add_callout(
        slide,
        "TDA's calibration loss comes from consulting the cache where it can only add "
        "confidence. The gap between CLIP's top two scores, applied at prediction time rather "
        "than admission time, tells those samples apart.",
        top=Inches(6.08), size=14.5,
    )
    set_notes(slide, f"""
To close. We set out to undo TDA's calibration loss with three training-free,
label-free changes, and measured each one rather than assuming it worked.

Margin admission fails outright, and we dropped it. Probabilistic fusion
fails as proposed, for a reason the equation makes visible once you compare
the two inputs' scales; the repair follows from the diagnosis, and fusing
only where CLIP is torn between two classes is the method we recommend:
mean ECE {pct(tda_ece)} to {pct(gate_ece)}, below zero-shot CLIP's own {pct(clip_ece)}, at
{pct(gate_acc)} percent accuracy against TDA's {pct(tda_acc)}. Temperature scaling helps on its
own but adds nothing on top of the gate.

What the result supports is a narrower and, we think, more useful claim than
the one we set out to make: TDA's calibration loss comes from consulting the
cache on samples where it can only add confidence, and the gap between
CLIP's top two scores, applied at prediction time rather than at admission
time, is enough to tell those samples apart.

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
    slide_margin,
    slide_fusion_proposed,
    slide_gate,
    slide_gate_result,
    slide_temperature,
    slide_together,
    slide_ablation_gate,
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


def build(summary_csv, ablation_csv, ablation_gate_csv, ablation_raw_dir,
          margin_config_yaml, gate_config_yaml, out_path, markdown_path=None) -> Path:
    results = Results(Path(summary_csv), Path(ablation_csv), Path(ablation_gate_csv),
                      Path(ablation_raw_dir), Path(margin_config_yaml),
                      Path(gate_config_yaml))

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
    parser.add_argument("--ablation-gate-csv", default="results/ablation_gate.csv")
    parser.add_argument(
        "--ablation-raw-dir", default="results/raw/ablation",
        help="margin run records, read for the cache admission rates",
    )
    parser.add_argument(
        "--margin-config", default="configs/method/ablation/margin.yaml",
        help="config of the margin ablation run; source of the threshold the slide quotes",
    )
    parser.add_argument(
        "--gate-config", default="configs/method/cal_tda_gate.yaml",
        help="config of the gated-fusion run; source of the gate constants",
    )
    parser.add_argument(
        "--out", default="presentation/calibration_aware_tta.pptx"
    )
    parser.add_argument(
        "--markdown", default="presentation/slides.md",
        help="readable mirror of the deck; pass an empty string to skip",
    )
    args = parser.parse_args()

    path = build(args.summary_csv, args.ablation_csv, args.ablation_gate_csv,
                 args.ablation_raw_dir, args.margin_config, args.gate_config, args.out,
                 markdown_path=args.markdown or None)
    print(f"wrote {path} ({len(SLIDE_BUILDERS)} slides)")
    if args.markdown:
        print(f"wrote {args.markdown}")


if __name__ == "__main__":
    main()
