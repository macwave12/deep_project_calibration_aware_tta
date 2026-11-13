"""Generate deck v5: agreement-gated TDA, diagnosis first (at most 20 main slides + backup).

Outline: `pm/research/T16-wrapup-plan.md` §10. Diagnosis and design slides use the val splits; results
slides use the test splits (and the development / fresh tiers); every figure says its split.

Every number on a slide or in the notes is READ from tracked files at build time and formatted through
`Results.num` (or `Results.illus` for the labelled illustrative values), which records it.
`tests/test_build_pptx.py` scans every slide and note for numerals and fails on any that did not pass
through those formatters, so a hand-typed number cannot reach the deck.

Sources: the tracked CSVs that `analysis/report_assets.py` reads (same numbers as the report),
`configs/method/cal_tda_agree.yaml` (alpha, beta, shot capacity), figures from
`analysis/paper_figures.py` and `analysis/story.py`, and the TDA paper's Fig. 1 and Fig. 2
(arXiv 2403.18293 source, `presentation/assets/tda_paper/`, cited on the slides).

    python analysis/paper_figures.py && python presentation/build_pptx.py

Writes `presentation/calibration_aware_tta.pptx` (16:9) and the speaker-notes mirror `presentation/slides.md`.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import yaml
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from analysis import report_assets as ra  # noqa: E402
from analysis.story import CONF_CUT, NAMES, SIGNAL_HEAD  # noqa: E402

ROOT = _ROOT
ASSETS = ROOT / "presentation" / "assets"
FIGURES = ROOT / "analysis" / "figures"
TDA_FIGS = ASSETS / "tda_paper"
CORE = ("dtd", "flower102", "pets", "eurosat", "aircraft")

# --- Theme ---------------------------------------------------------------------
INK = RGBColor(0x1A, 0x1F, 0x2B)
MUTED = RGBColor(0x55, 0x5C, 0x6B)
ACCENT = RGBColor(0x7D, 0x5B, 0xA6)       # ours
BASELINE = RGBColor(0xDD, 0x84, 0x52)     # TDA
CLIPBLUE = RGBColor(0x4C, 0x72, 0xB0)     # CLIP
NEUTRAL = RGBColor(0xC9, 0xC6, 0xC0)
RULE = RGBColor(0xD8, 0xD5, 0xD0)
SURFACE = RGBColor(0xFF, 0xFF, 0xFF)
WARN = RGBColor(0xB3, 0x4A, 0x3C)
GOOD = RGBColor(0x3C, 0x8C, 0x5A)
BG = RGBColor(0xF4, 0xF0, 0xE9)
BG_DARK = RGBColor(0x25, 0x1F, 0x36)
HEADER_FILL = RGBColor(0xE7, 0xE2, 0xD8)
TINT_ACCENT = RGBColor(0xEC, 0xE6, 0xF3)
TINT_WARN = RGBColor(0xF7, 0xEA, 0xE1)
TINT_CLIP = RGBColor(0xE4, 0xEB, 0xF5)

FONT = "Calibri"
BODY = 20          # main bullets
SUB = 18           # sub-bullets, panel text: the floor for body text
TABLE = 14         # table text floor
SMALL = 14         # source lines and figure captions only

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)
MARGIN = Inches(0.6)
BODY_TOP = Inches(1.62)
BODY_BOTTOM = Inches(6.85)
CONTENT_W = SLIDE_W - 2 * MARGIN

# Numerals recognised as data on a slide: a digit not glued to a word, a slash,
# an apostrophe or a hyphen (so "ViT-B/16", "CVPR'24", "A3", "top-2", "16-shot"
# and "RN50" are names, not numbers). The test uses the same pattern.
NUMBER_RE = re.compile(r"(?<![\w'/.\-])\d[\d,]*(?:\.\d+)?(?![\w'/\-])")


def number_tokens(text: str) -> list[str]:
    return [token.rstrip(",") for token in NUMBER_RE.findall(text)]


# --- Data ----------------------------------------------------------------------
class Results:
    """Every number the deck shows, read from tracked files, formatted and recorded."""

    def __init__(self, root: Path = ROOT):
        root = Path(root)
        self.d = ra.load(root)
        self.cfg = yaml.safe_load((root / "configs/method/cal_tda_agree.yaml").read_text(encoding="utf-8"))
        self.gen = ra.generalization(self.d)
        self.emitted: set[str] = set()
        self.illustrative: set[str] = set()

    def _record(self, text: str, bucket: set) -> str:
        bucket.update(number_tokens(text))
        return text

    def num(self, value, digits=2, *, sign=False, comma=False) -> str:
        """Format a value read from data and record it as legitimately on the deck."""
        if comma:
            text = f"{value:,.0f}"
        elif sign:
            text = f"{value:+.{digits}f}".replace("-", "−")
        else:
            text = f"{value:.{digits}f}".replace("-", "−")
        return self._record(text, self.emitted)

    def illus(self, value, digits=2) -> str:
        text = f"{value:.{digits}f}"
        return self._record(text, self.illustrative)

    # lookups (P2 throughout) --------------------------------------------------
    def main(self, method, metric, dataset="mean", col="mean"):
        return ra.prot(self.d, dataset, method, metric, col)

    def pair(self, method, metric, reference="tda", dataset="mean"):
        return ra.paired(self.d, method, metric, reference, dataset)

    def comp(self, method, metric, dataset="mean"):
        return ra.comp(self.d, method, metric, dataset)

    def comp_pair(self, method, metric, dataset="mean"):
        return ra.comp_paired(self.d, method, metric, dataset)

    def val(self, column, dataset="mean"):
        t = self.d["val"]
        return float(t[t.dataset == dataset].iloc[0][column])

    def test(self, column, dataset="mean"):
        t = self.d["test"]
        return float(t[t.dataset == dataset].iloc[0][column])

    def rho(self, signal="conf", datasets="all"):
        t = self.d["rho"]
        return t[(t.signal == signal) & (t.datasets == datasets)].iloc[0]



# --- Low-level drawing helpers ---------------------------------------------------
def _style(run, size, *, bold=False, color=INK, italic=False, font=FONT):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    run.font.name = font


def textbox(slide, left, top, width, height, *, anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(int(left), int(top), int(width), int(height))
    frame = box.text_frame
    frame.word_wrap = True
    frame.vertical_anchor = anchor
    frame.margin_left = frame.margin_right = 0
    frame.margin_top = frame.margin_bottom = 0
    return box, frame


def add_text(slide, left, top, width, height, paragraphs, *, anchor=MSO_ANCHOR.TOP,
             align=PP_ALIGN.LEFT, spacing=6):
    """`paragraphs`: list of (text, size, bold, color) or lists of such runs."""
    _, frame = textbox(slide, left, top, width, height, anchor=anchor)
    for index, para in enumerate(paragraphs):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.alignment = align
        if index:
            paragraph.space_before = Pt(spacing)
        runs = para if isinstance(para, list) else [para]
        for text, size, bold, color in runs:
            run = paragraph.add_run()
            run.text = text
            _style(run, size, bold=bold, color=color)
    return frame


def paint_background(slide, color=BG):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = color
    band = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, Inches(0.09))
    band.fill.solid()
    band.fill.fore_color.rgb = ACCENT
    band.line.fill.background()
    band.shadow.inherit = False


def new_slide(prs, title, *, kicker=None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    paint_background(slide)
    top = Inches(0.36)
    if kicker:
        _, frame = textbox(slide, MARGIN, top, CONTENT_W, Inches(0.3))
        run = frame.paragraphs[0].add_run()
        run.text = kicker.upper()
        _style(run, 14, bold=True, color=ACCENT)
        top += Inches(0.34)
    box, frame = textbox(slide, MARGIN, top, CONTENT_W, Inches(0.62))
    box.name = "title"
    run = frame.paragraphs[0].add_run()
    run.text = title
    _style(run, 30, bold=True, color=INK)
    return slide


def add_bullets(slide, items, *, left=None, top=None, width=None, height=None,
                size=BODY, sub_size=SUB, spacing=8):
    """`items`: strings, (level, text) or (level, text, color)."""
    left = MARGIN if left is None else left
    top = BODY_TOP if top is None else top
    width = CONTENT_W if width is None else width
    height = (BODY_BOTTOM - top) if height is None else height
    _, frame = textbox(slide, left, top, width, height)
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
        paragraph.space_after = Pt(spacing if level == 0 else spacing * 0.6)
        if not text:
            _style(paragraph.add_run(), 8)
            continue
        hang = Inches(0.30)
        properties = paragraph._p.get_or_add_pPr()
        properties.set("marL", str(int(hang if level == 0 else hang + Inches(0.42))))
        properties.set("indent", str(int(-hang)))
        run = paragraph.add_run()
        run.text = ("•   " if level == 0 else "–   ") + text
        _style(run, size if level == 0 else sub_size, color=color if level == 0 else MUTED)
    return frame


def add_panel(slide, left, top, width, height, *, fill=SURFACE, line=RULE, dashed=False):
    panel = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, int(left), int(top), int(width), int(height))
    panel.fill.solid()
    panel.fill.fore_color.rgb = fill
    if line is None:
        panel.line.fill.background()
    else:
        panel.line.color.rgb = line
        panel.line.width = Pt(1.5 if dashed else 1.25)
        if dashed:
            panel.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    panel.shadow.inherit = False
    panel.adjustments[0] = 0.05
    panel.text_frame.text = ""
    return panel


def add_callout(slide, text, *, top=Inches(6.18), color=ACCENT, size=SUB, left=None, width=None,
                height=Inches(0.62)):
    left = MARGIN if left is None else left
    width = CONTENT_W if width is None else width
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, int(left), int(top), Pt(5), int(height))
    bar.fill.solid()
    bar.fill.fore_color.rgb = color
    bar.line.fill.background()
    bar.shadow.inherit = False
    _, frame = textbox(slide, left + Inches(0.2), top, width - Inches(0.2), height, anchor=MSO_ANCHOR.MIDDLE)
    run = frame.paragraphs[0].add_run()
    run.text = text
    _style(run, size, bold=True, color=INK)
    return frame


def add_placeholder(slide, left, top, width, height, evidence, what, line):
    """A clearly styled box where pending evidence will go."""
    add_panel(slide, left, top, width, height, fill=PENDING_FILL, line=PENDING_LINE, dashed=True)
    pad = Inches(0.18)
    add_text(slide, left + pad, top + pad, width - 2 * pad, height - 2 * pad, [
        (f"[Pending {evidence}: {what}]", SUB, True, PENDING_LINE),
        (line, SUB, False, INK),
    ], anchor=MSO_ANCHOR.MIDDLE, spacing=4)


def add_picture(slide, path, left, top, max_width, max_height, *, align="center"):
    picture = slide.shapes.add_picture(str(path), int(left), int(top), width=int(max_width))
    if picture.height > max_height:
        picture._element.getparent().remove(picture._element)
        picture = slide.shapes.add_picture(str(path), int(left), int(top), height=int(max_height))
    if align == "center":
        picture.left = int(left + (max_width - picture.width) / 2)
    picture.top = int(top + (max_height - picture.height) / 2)
    return picture


def add_table(slide, header, rows, left, top, width, *, col_widths=None, font_size=TABLE,
              row_height=Inches(0.4), highlight_rows=(), first_col_align=PP_ALIGN.LEFT,
              align=PP_ALIGN.RIGHT, bold_first_col=False, cell_colors=None, aligns=None):
    """Text columns can be left-aligned via `aligns` ("L"/"R" per column)."""
    def column_align(col):
        if aligns:
            return PP_ALIGN.LEFT if aligns[col] == "L" else PP_ALIGN.RIGHT
        return first_col_align if col == 0 else align

    shape = slide.shapes.add_table(len(rows) + 1, len(header), int(left), int(top), int(width),
                                   int(row_height * (len(rows) + 1)))
    table = shape.table
    table.first_row = True
    table.horz_banding = False
    if col_widths:
        total = sum(col_widths)
        for index, fraction in enumerate(col_widths):
            table.columns[index].width = Emu(int(width * fraction / total))
    for col, text in enumerate(header):
        cell = table.cell(0, col)
        cell.text = ""
        cell.fill.solid()
        cell.fill.fore_color.rgb = HEADER_FILL
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        cell.margin_left = cell.margin_right = Inches(0.07)
        cell.margin_top = cell.margin_bottom = Inches(0.03)
        paragraph = cell.text_frame.paragraphs[0]
        paragraph.alignment = column_align(col)
        run = paragraph.add_run()
        run.text = text
        _style(run, font_size, bold=True, color=MUTED)
    table.rows[0].height = int(row_height)
    for r_index, row in enumerate(rows, start=1):
        table.rows[r_index].height = int(row_height)
        emphasised = (r_index - 1) in highlight_rows
        for col, text in enumerate(row):
            cell = table.cell(r_index, col)
            cell.text = ""
            cell.fill.solid()
            cell.fill.fore_color.rgb = TINT_ACCENT if emphasised else SURFACE
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left = cell.margin_right = Inches(0.07)
            cell.margin_top = cell.margin_bottom = Inches(0.03)
            paragraph = cell.text_frame.paragraphs[0]
            paragraph.alignment = column_align(col)
            run = paragraph.add_run()
            run.text = text
            color = INK
            if cell_colors and (r_index - 1, col) in cell_colors:
                color = cell_colors[(r_index - 1, col)]
            _style(run, font_size, bold=emphasised or (bold_first_col and col == 0), color=color)
    return table


def flow_box(slide, left, top, width, height, title, subtitle=None, *, fill=SURFACE, border=RULE,
             title_size=SUB, subtitle_size=SUB, title_color=INK):
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, int(left), int(top), int(width), int(height))
    box.fill.solid()
    box.fill.fore_color.rgb = fill
    box.line.color.rgb = border
    box.line.width = Pt(1.5)
    box.shadow.inherit = False
    box.adjustments[0] = 0.08
    frame = box.text_frame
    frame.word_wrap = True
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    frame.margin_left = frame.margin_right = Inches(0.1)
    frame.margin_top = frame.margin_bottom = Inches(0.05)
    paragraph = frame.paragraphs[0]
    paragraph.alignment = PP_ALIGN.CENTER
    run = paragraph.add_run()
    run.text = title
    _style(run, title_size, bold=True, color=title_color)
    if subtitle:
        paragraph = frame.add_paragraph()
        paragraph.alignment = PP_ALIGN.CENTER
        paragraph.space_before = Pt(4)
        run = paragraph.add_run()
        run.text = subtitle
        _style(run, subtitle_size, color=MUTED)
    return box


def arrow(slide, left, top, width, height, shape=MSO_SHAPE.RIGHT_ARROW, color=NEUTRAL):
    a = slide.shapes.add_shape(shape, int(left), int(top), int(width), int(height))
    a.fill.solid()
    a.fill.fore_color.rgb = color
    a.line.fill.background()
    a.shadow.inherit = False
    a.text_frame.text = ""
    return a


def dot(slide, cx, cy, d, color, *, outline=None, shape=MSO_SHAPE.OVAL):
    s = slide.shapes.add_shape(shape, int(cx - d / 2), int(cy - d / 2), int(d), int(d))
    s.fill.solid()
    s.fill.fore_color.rgb = color
    if outline is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = outline
        s.line.width = Pt(2.25)
    s.shadow.inherit = False
    s.text_frame.text = ""
    return s


def add_footer(slide, index, total, label=None):
    box, frame = textbox(slide, MARGIN, Inches(7.06), Inches(8.0), Inches(0.3))
    box.name = "footer"
    run = frame.paragraphs[0].add_run()
    run.text = "Calibration-aware TTA for CLIP  ·  deck v5"
    _style(run, 12, color=MUTED)
    box, frame = textbox(slide, SLIDE_W - MARGIN - Inches(1.4), Inches(7.06), Inches(1.4), Inches(0.3))
    box.name = "footer"
    frame.paragraphs[0].alignment = PP_ALIGN.RIGHT
    run = frame.paragraphs[0].add_run()
    run.text = label or f"{index} / {total}"
    _style(run, 12, color=MUTED)


def set_notes(slide, text):
    """Notes as paragraphs: blank lines separate paragraphs, single newlines are rewrapped."""
    paragraphs = [" ".join(p.split()) for p in text.strip().split("\n\n") if p.strip()]
    slide.notes_slide.notes_text_frame.text = "\n\n".join(paragraphs)


# --- Rendered assets (equations and small illustrative figures) -----------------
def equation(tex: str, *, size=24, color="#1A1F2B") -> Path:
    """Render a mathtext equation to a transparent PNG at its true point size."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    key = hashlib.sha1(f"{tex}|{size}|{color}".encode()).hexdigest()[:12]
    path = ASSETS / f"eq_{key}.png"
    if path.exists():
        return path
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f"${tex}$", fontsize=size, color=color)
    fig.savefig(path, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return path


def add_equation(slide, tex, left, top, *, size=24, color="#1A1F2B", center_width=None):
    path = equation(tex, size=size, color=color)
    from PIL import Image
    with Image.open(path) as image:
        w_px, h_px = image.size
    width, height = Inches(w_px / 300), Inches(h_px / 300)
    if center_width is not None:
        if width > center_width:   # never let an equation overflow its column
            height = int(height * center_width / width)
            width = center_width
        left = left + (center_width - width) / 2
    slide.shapes.add_picture(str(path), int(left), int(top), width=int(width), height=int(height))
    return height



# --- Slides -----------------------------------------------------------------------
# Each builder: (prs, r) -> slide. Notes are spoken English; numbers via r.num.

def slide_title(prs, r):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = BG_DARK
    band = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, SLIDE_H - Inches(0.14), SLIDE_W, Inches(0.14))
    band.fill.solid()
    band.fill.fore_color.rgb = ACCENT
    band.line.fill.background()
    light, lavender, soft = RGBColor(0xF5, 0xF2, 0xFA), RGBColor(0xB9, 0xA6, 0xD6), RGBColor(0xC8, 0xC2, 0xD8)
    add_text(slide, Inches(1.1), Inches(1.6), Inches(11.2), Inches(0.4),
             [("TAU  ·  DEEP LEARNING  ·  FINAL PROJECT", 16, True, lavender)])
    add_text(slide, Inches(1.1), Inches(2.15), Inches(11.2), Inches(1.8),
             [("The cache that echoes CLIP: agreement-gated test-time adaptation", 40, True, light)])
    add_text(slide, Inches(1.1), Inches(4.25), Inches(11.2), Inches(1.6), [
        ("Gal Machlin   ·   Idan Yehuda", 22, False, light),
        ("Why TDA's accuracy gain makes CLIP overconfident, a proof of where it happens, "
         "and a parameter-free fix.", 20, False, soft),
    ], spacing=14)
    set_notes(slide, """
Hello everyone. This talk is about test-time adaptation of CLIP, and in particular about TDA, a
popular training-free method that improves CLIP's accuracy but makes its confidence less
trustworthy. We first diagnose exactly where that comes from, prove it, and then show a fix
that needs no parameters and keeps TDA's accuracy while halving its calibration error.
""")
    return slide


def slide_agenda(prs, r):
    slide = new_slide(prs, "Roadmap", kicker="Overview")
    entries = [
        ("1", "The task and CLIP", "zero-shot classification of an unlabeled test stream"),
        ("2", "Test-time adaptation and TDA", "episodic vs online methods; how TDA's caches work"),
        ("3", "Calibration and TDA's catch", "what ECE measures; TDA gains accuracy but becomes overconfident"),
        ("4", "Diagnosis and method", "the cache echoes CLIP (with a proof); the agreement gate"),
        ("5", "Results and limits", "test sets, baselines, unseen datasets, where it helps, what agreement cannot tell"),
    ]
    top = Inches(1.8)
    for index, (number, heading, detail) in enumerate(entries):
        row_top = top + index * Inches(0.9)
        chip = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, MARGIN, int(row_top), Inches(0.62), Inches(0.62))
        chip.fill.solid()
        chip.fill.fore_color.rgb = ACCENT if index == 3 else HEADER_FILL
        chip.line.fill.background()
        chip.shadow.inherit = False
        chip.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = chip.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = number
        _style(run, 20, bold=True, color=SURFACE if index == 3 else MUTED)
        add_text(slide, MARGIN + Inches(0.9), row_top, Inches(11.0), Inches(0.62), [[
            (heading, 22, True, INK), ("      " + detail, SUB, False, MUTED)]], anchor=MSO_ANCHOR.MIDDLE)
    set_notes(slide, """
Five parts. First the task: classifying images with CLIP when we only know the class names and the
test images arrive one by one without labels. Second, test-time adaptation, the family of methods that
uses that stream to adapt, and TDA, the method we study, in some detail. Third, calibration, and why TDA
makes CLIP overconfident. Fourth, our diagnosis of where that comes from, with a short proof, and the
fix it suggests. Fifth, the results, including where the fix does not help.
""")
    return slide


def slide_task(prs, r):
    slide = new_slide(prs, "The task: classify a stream with nothing but the class names", kicker="1 · Task and CLIP")
    add_bullets(slide, [
        "Given: the names of the classes (e.g. the breeds of Oxford Pets) and a pretrained vision-language model.",
        "Given: unlabeled test images that arrive one at a time.",
        "Not given: any labeled image, for training or for tuning.",
        "Each prediction is final before the next image arrives (online), and should come with an honest "
        "confidence: right as often as it claims.",
    ], width=Inches(7.3), spacing=14)
    add_panel(slide, Inches(8.25), Inches(1.75), Inches(4.48), Inches(3.9))
    add_text(slide, Inches(8.5), Inches(1.95), Inches(4.0), Inches(3.6), [
        ("Our evaluation", SUB, True, ACCENT),
        ("CLIP ViT-B/16, one prompt template", SUB, False, INK),
        ("Five core datasets: DTD, Flowers, Pets, EuroSAT, Aircraft (val splits for design, test splits for results)",
         SUB, False, INK),
        ("Development and fresh datasets for generalization (ImageNet and others)", SUB, False, INK),
        ("Each stream order is a seed", SUB, False, INK),
    ], spacing=10)
    add_callout(slide, "Zero-shot: no labels anywhere. Labels are read only by the final scorer.")
    set_notes(slide, """
Let me state the task precisely. We are given the class names and a pretrained model, CLIP, and a
stream of test images without labels. There is no labeled data at all, not even for tuning. Each image
gets a prediction before the next one arrives. And we want two things: the right class, and a
confidence we can believe. On the right is how we evaluate: five core datasets whose validation splits
we used for design and whose test splits we use for results, plus further datasets for generalization.
""")
    return slide


def slide_clip(prs, r):
    slide = new_slide(prs, "Zero-shot CLIP: compare the image with a prompt per class", kicker="1 · Task and CLIP")
    top1, top2, h = Inches(1.8), Inches(3.25), Inches(1.0)
    flow_box(slide, MARGIN, top1, Inches(2.3), h, "test image x", fill=SURFACE)
    arrow(slide, MARGIN + Inches(2.38), top1 + Inches(0.34), Inches(0.5), Inches(0.32))
    flow_box(slide, MARGIN + Inches(2.95), top1, Inches(3.3), h, "image encoder f", "frozen",
             fill=TINT_CLIP, border=CLIPBLUE)
    flow_box(slide, MARGIN, top2, Inches(2.3), h, "class name k", fill=SURFACE)
    arrow(slide, MARGIN + Inches(2.38), top2 + Inches(0.34), Inches(0.5), Inches(0.32))
    flow_box(slide, MARGIN + Inches(2.95), top2, Inches(3.3), h, "text encoder g", "frozen",
             fill=TINT_CLIP, border=CLIPBLUE)
    add_text(slide, MARGIN, top2 + h + Inches(0.12), Inches(6.3), Inches(0.45),
             [("prompt:  “a photo of a {class}.”", SUB, True, ACCENT)])
    arrow(slide, MARGIN + Inches(6.35), top1 + Inches(1.05), Inches(0.5), Inches(0.4))
    add_panel(slide, MARGIN + Inches(7.0), top1, Inches(5.1), Inches(2.45), fill=SURFACE)
    add_equation(slide, r"z_k = s\,\cos\left(f(x),\,g(t_k)\right)", MARGIN + Inches(7.1), top1 + Inches(0.3),
                 size=26, center_width=Inches(4.9))
    add_equation(slide, r"p = \mathrm{softmax}(z),\quad \hat{y}_0 = \arg\max_k\, z_k",
                 MARGIN + Inches(7.1), top1 + Inches(1.35), size=26, center_width=Inches(4.9))
    add_bullets(slide, [
        "Both encoders stay frozen; every method in this talk only changes what is done with features and logits.",
        "The prediction ŷ₀ is CLIP's pseudo-label; its probability max p is the confidence.",
        "It works without labels, but loses accuracy when the test images look unlike CLIP's web data "
        "(satellite tiles, textures) or the classes are fine-grained (aircraft variants).",
    ], top=Inches(4.75), spacing=6)
    set_notes(slide, """
This is the model everything builds on. CLIP has an image encoder and a text encoder, both frozen. Each
class name is put into a prompt and encoded; the logit of a class is the cosine between the image
feature and that text feature, times a learned scale. The softmax gives a distribution, its argmax is
the prediction, which we will call CLIP's pseudo-label, and its probability is the confidence. Zero-shot
CLIP needs no labels, but it is weak when the images look unlike web photos or the classes are very
fine-grained. That is what test-time adaptation tries to fix.
""")
    return slide


def render_calibration_figure(out_path: Path) -> Path:
    """Textbook reliability diagram from SYNTHETIC curves (concept only)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    confidence = np.linspace(0.30, 0.975, 14)
    over = confidence ** 2.4
    fig, ax = plt.subplots(figsize=(4.6, 4.3))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#4D4D4D", linewidth=1.8, label="calibrated")
    ax.plot(confidence, over, marker="o", markersize=6, color="#B34A3C", linewidth=2.2, label="overconfident")
    pick = 10
    ax.plot([confidence[pick]] * 2, [over[pick], confidence[pick]], color="#7D5BA6", linewidth=3.2,
            label="one bin's gap")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("confidence (bin mean)", fontsize=14)
    ax.set_ylabel("accuracy in the bin", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=13, frameon=True, framealpha=0.95, edgecolor="#D8D5D0")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def slide_calibration(prs, r):
    slide = new_slide(prs, "Calibration: confidence should equal accuracy", kicker="3 · Calibration")
    add_panel(slide, MARGIN, Inches(1.65), Inches(7.0), Inches(1.4))
    add_equation(slide, r"\mathrm{ECE} = \sum_{b} \frac{|B_b|}{N}\,\left|\,\mathrm{acc}(B_b) - \mathrm{conf}(B_b)\,\right|",
                 MARGIN, Inches(1.85), size=26, center_width=Inches(7.0))
    add_bullets(slide, [
        "Bin the predictions by confidence; in each bin compare the mean confidence with the accuracy.",
        "ECE is the sample-weighted mean gap (equal-width bins, as the benchmark). Lower is better.",
        f"Overconfident: confidence above accuracy. Illustrative: a model that says {r.illus(90, 0)}% should be "
        f"right {r.illus(90, 0)}% of the time.",
        "We also report the per-sample gap confidence − accuracy on subsets of the stream.",
    ], top=Inches(3.3), width=Inches(7.0), spacing=10)
    path = render_calibration_figure(ASSETS / "what_is_calibration.png")
    add_picture(slide, path, Inches(8.0), Inches(1.6), Inches(4.7), Inches(4.4))
    add_text(slide, Inches(8.0), Inches(6.05), Inches(4.7), Inches(0.4), [
        ("Concept sketch with synthetic curves, not data.", SMALL, False, MUTED)], align=PP_ALIGN.CENTER)
    set_notes(slide, """
Calibration means the confidence a model reports matches how often it is right. The expected
calibration error bins the predictions by confidence and averages, over the bins, the gap between
mean confidence and accuracy, weighted by the number of samples in each bin. The sketch on the right
is a concept drawing, not data: points below the diagonal are overconfident. Throughout the talk we
also use the signed gap, confidence minus accuracy, on parts of the stream.
""")
    return slide


# --- TDA section ---------------------------------------------------------------------------
TDA_CITE = "Figure: Karmanov et al., Efficient Test-Time Adaptation of Vision-Language Models, CVPR'24, arxiv.org/abs/2403.18293."


def slide_tda_why(prs, r):
    slide = new_slide(prs, "Two ways to adapt CLIP at test time", kicker="2 · Test-time adaptation")
    add_picture(slide, TDA_FIGS / "Prompt-Tuning.png", MARGIN, Inches(1.6), Inches(6.2), Inches(2.3))
    add_picture(slide, TDA_FIGS / "TDA.png", MARGIN, Inches(3.95), Inches(6.2), Inches(2.4))
    add_bullets(slide, [
        "Episodic, with backprop (top: TPT, DiffTPT; also C-TPT, O-TPT): for each image, many augmented views "
        "and gradient steps on the prompt, then reset. Slow; nothing carries over.",
        "Online, training-free (bottom: TDA; also DMN, ECALP): keep a memory of past test images and use it "
        "for the next ones. One forward pass per image.",
        "We study TDA (CVPR'24): the standard online baseline, and later cache methods build on it.",
    ], left=Inches(7.1), top=Inches(1.7), width=Inches(5.6), spacing=12)
    add_text(slide, MARGIN, Inches(6.45), CONTENT_W, Inches(0.4), [("TDA paper, Fig. 1. " + TDA_CITE, SMALL, False, MUTED)])
    set_notes(slide, """
Test-time adaptation uses the unlabeled stream to improve CLIP. There are two families. This is the
first figure of the TDA paper. On top, test-time prompt tuning: for every image it builds many augmented
views and optimizes the prompt by backpropagation, then resets, which is slow. Below, TDA: no training,
only a cache of earlier test images labelled by CLIP, used for the next images. It is fast and fully
online, which is why it became the standard baseline of the training-free family and why we study it.
""")
    return slide


def slide_tda_pipeline(prs, r):
    slide = new_slide(prs, "TDA's two caches", kicker="2 · TDA")
    add_picture(slide, TDA_FIGS / "TDA-full.png", MARGIN, Inches(1.55), CONTENT_W, Inches(3.6))
    add_bullets(slide, [
        "Positive cache: per class, the lowest-entropy test images under CLIP's pseudo-label; votes for classes.",
        "Negative cache: mid-entropy images with a mask of unlikely classes; votes against classes.",
        "The values (labels) always come from CLIP's prediction, never from TDA's output.",
    ], top=Inches(5.25), spacing=6, size=SUB)
    add_text(slide, MARGIN, Inches(6.5), CONTENT_W, Inches(0.35), [("TDA paper, Fig. 2. " + TDA_CITE, SMALL, False, MUTED)])
    set_notes(slide, """
This is TDA's pipeline, from its paper. CLIP produces a prediction. Two caches store test features
as keys and pseudo-labels as values: a positive cache of confident samples, and a negative cache of
uncertain samples that records which classes are unlikely. Both caches produce predictions that are
added to CLIP's. The important detail for us is the update path at the bottom: the cache values are
CLIP's own predictions.
""")
    return slide


def slide_tda_math(prs, r):
    slide = new_slide(prs, "TDA in one equation", kicker="2 · TDA")
    a, b, s = r.num(r.cfg["alpha"], 0), r.num(r.cfg["beta"], 0), r.num(r.cfg["shot_capacity"], 0)
    add_panel(slide, MARGIN, Inches(1.65), CONTENT_W, Inches(1.65))
    add_equation(slide, r"o(x) = z(x) - n(x) + \alpha\,c(x), \qquad "
                 r"c_k(x) = \sum_{i \in \mathcal{C}_k} e^{-\beta\,(1 - \langle f(x), f_i\rangle)} \in [0, S]",
                 MARGIN, Inches(1.85), size=26, center_width=CONTENT_W)
    add_bullets(slide, [
        "z: CLIP's logits; ŷ₀ = argmax z is the pseudo-label written to the cache.",
        "n: the negative-cache term; b = z − n is CLIP plus the negative cache.",
        f"c: the positive cache's kernel votes, at most S = {s} items per class, each vote at most 1.",
        "Each image is first written to the cache (if admitted) and then read.",
        f"One setting for every dataset (our protocol P2): α = {a}, β = {b}. The published per-dataset "
        "values (P1) give the same conclusions (backup).",
    ], top=Inches(3.55), spacing=10)
    set_notes(slide, f"""
Here is TDA in one line. The output logits are CLIP's logits, minus the negative cache term, plus
alpha times the positive cache's votes. Each positive cache item votes with a kernel of its cosine to
the test image, so each vote is at most one, and with {s} items per class a class gets at most {s}.
Two facts matter later: the cache is filled with CLIP's argmax, and each image is written before it
is read. We use alpha {a} and beta {b} on every dataset.
""")
    return slide


def slide_tda_caches(prs, r):
    slide = new_slide(prs, "Which cache does what (TDA paper's Fig. 3, with ECE)", kicker="3 · TDA\'s catch")
    rows = []
    for k, label in (("clip", "CLIP"), ("base", "CLIP + negative cache"), ("pos", "CLIP + positive cache"),
                     ("tda", "TDA (both)")):
        rows.append([label, r.num(r.val(f"{k}_acc")), r.num(r.val(f"{k}_ece"))])
    add_table(slide, ["variant (val, mean of 5)", "accuracy", "ECE"], rows, MARGIN, Inches(1.75), Inches(6.6),
              col_widths=[3.2, 1.2, 1.2], row_height=Inches(0.5), font_size=18, highlight_rows=(3,))
    add_bullets(slide, [
        "The negative cache adds accuracy and lowers ECE.",
        "The positive cache adds more accuracy, and all of the overconfidence.",
        "So the problem sits in the positive cache. The next section asks why.",
    ], left=Inches(7.6), top=Inches(1.8), width=Inches(5.1), spacing=14)
    add_callout(slide, "Exact re-evaluation of one TDA run per seed (caches do not depend on α). Val splits.")
    set_notes(slide, f"""
The TDA paper ablates its two caches for accuracy. We repeat that with calibration, on the val
splits. CLIP alone has {r.num(r.val('clip_acc'))} percent accuracy and ECE {r.num(r.val('clip_ece'))}. The negative cache alone
improves both. The positive cache brings accuracy, {r.num(r.val('pos_acc'))} percent, but its ECE is {r.num(r.val('pos_ece'))}. Full TDA ends
at {r.num(r.val('tda_acc'))} and ECE {r.num(r.val('tda_ece'))}. The overconfidence comes from the positive cache.
""")
    return slide


def slide_tda_overconfident(prs, r):
    slide = new_slide(prs, "TDA's catch: more accuracy, worse calibration", kicker="3 · TDA\'s catch")
    rows = []
    for ds in CORE + ("mean",):
        rows.append([NAMES.get(ds, "Mean"), r.num(r.main("clipzs", "accuracy", ds)), r.num(r.main("tda", "accuracy", ds)),
                     r.num(r.main("clipzs", "ece", ds)), r.num(r.main("tda", "ece", ds))])
    add_table(slide, ["dataset (test)", "CLIP acc.", "TDA acc.", "CLIP ECE", "TDA ECE"], rows, MARGIN, Inches(1.7),
              Inches(7.4), col_widths=[2.0, 1.2, 1.2, 1.2, 1.2], row_height=Inches(0.48), font_size=17,
              highlight_rows=(5,))
    add_bullets(slide, [
        "TDA gains accuracy on every dataset.",
        "ECE rises on most, sharply on DTD, Flowers and Aircraft.",
        "Pets is the exception: CLIP is underconfident there, and TDA's extra confidence happens to help.",
    ], left=Inches(8.4), top=Inches(1.8), width=Inches(4.3), spacing=14)
    add_callout(slide, "The benchmark (Illusion of Progress, NeurIPS'25) finds the same for TTA methods in general. "
                "Our question: why, and can it be fixed at the source?")
    seeds = r.num(r.pair("cal_tda_agree", "ece").n, 0)
    set_notes(slide, f"""
Per dataset, on the test splits with {seeds} stream orders. TDA gains accuracy everywhere, but the
calibration error grows on most datasets; on DTD it goes from {r.num(r.main('clipzs', 'ece', 'dtd'))} to {r.num(r.main('tda', 'ece', 'dtd'))}.
Pets is the exception: CLIP is underconfident on Pets and TDA's extra confidence happens to correct
it. Keep Pets in mind, it will be the one dataset where our method is worse.
""")
    return slide


# --- Diagnosis ---------------------------------------------------------------------------
def slide_selfvote(prs, r):
    slide = new_slide(prs, "Symptom: the self-vote", kicker="4 · Diagnosis · val")
    v = r.d["val"][r.d["val"].dataset != "mean"]
    lo, hi = r.num(v.self_vote_share.min(), 0), r.num(v.self_vote_share.max(), 0)
    add_picture(slide, FIGURES / "paper_selfvote_val.png", MARGIN, Inches(1.55), CONTENT_W, Inches(3.5))
    add_bullets(slide, [
        "Write before read: an admitted image finds itself in the cache, with cosine 1 and vote exp(0) = 1, "
        "for its own pseudo-label.",
        f"Frequent ({lo}–{hi}% of the stream), and it makes the cache agree with CLIP almost always.",
        f"Removing only the self term lowers TDA's val ECE from {r.num(r.val('tda_ece'))} to {r.num(r.val('noself_ece'))}. "
        "But the self-vote is only the most visible part of a broader echo.",
    ], top=Inches(5.15), spacing=6, size=SUB)
    set_notes(slide, f"""
The first symptom. Because each image is written before it is read, an admitted image votes for its
own pseudo-label with weight exactly one. This carries no information about the true label. It
happens for {lo} to {hi} percent of the stream on the val splits, and when it happens the cache almost
always agrees with CLIP, left panel. Removing just that term lowers TDA's ECE from {r.num(r.val('tda_ece'))} to
{r.num(r.val('noself_ece'))}, right panel. So the self-vote matters, but it is not the whole story.
""")
    return slide


def slide_lemmas(prs, r):
    slide = new_slide(prs, "Where the cache agrees with CLIP, it can only add confidence", kicker="4 · Diagnosis · proof")
    add_panel(slide, MARGIN, Inches(1.6), CONTENT_W, Inches(2.25), fill=TINT_ACCENT, line=ACCENT)
    add_text(slide, MARGIN + Inches(0.25), Inches(1.7), Inches(11.6), Inches(0.4),
             [("Lemma (agreement). If argmax b = argmax c = ŷ, then", SUB, True, INK)])
    add_equation(slide, r"o_{\hat y} - o_j = (b_{\hat y} - b_j) + \alpha\,(c_{\hat y} - c_j) \;\geq\; b_{\hat y} - b_j > 0",
                 MARGIN, Inches(2.2), size=26, center_width=CONTENT_W)
    add_text(slide, MARGIN + Inches(0.25), Inches(3.1), Inches(11.6), Inches(0.6),
             [("so the decision is unchanged, every margin grows, and max softmax(o) ≥ max softmax(b).", SUB, False, INK)])
    add_bullets(slide, [
        "The gate tests argmax c = CLIP's top class; the lemma needs argmax c = argmax b. They differ only where the "
        f"negative cache moved CLIP's top class: {r.num(r.val('agree_neg_moved'), 1)}% of agreeing val samples "
        f"(TDA flips {r.num(r.val('agree_tda_flips'), 1)}%, with no net accuracy gain). Agreement mostly adds confidence.",
        "No-flip region: if CLIP's top-2 logit gap exceeds αS + range(n), nothing in the cache can change the "
        "decision (0 ≤ c ≤ S). Cf. MARC, arxiv.org/abs/2609.23409.",
        "The cache is filled with CLIP's own argmax, so agreement is the common case: an echo.",
    ], top=Inches(4.1), spacing=10)
    set_notes(slide, """
Now the formal argument. Let b be CLIP plus the negative cache. If the positive cache's top class is
the same as b's top class, then every margin of the output is the margin of b plus alpha times a
non-negative cache margin. So the decision cannot change, and the top probability can only increase.
The gate checks agreement with CLIP's own top class, which differs from the lemma's condition only
where the negative cache moved that top class, a small share of samples. So where the cache agrees it
mostly adds confidence, and the accuracy gain comes from the samples where it disagrees. Since the
cache is filled with CLIP's own labels, agreement is the common case: the cache echoes CLIP.
""")
    return slide


def slide_agreement(prs, r):
    slide = new_slide(prs, "Backup: the echo measured on val, by agreement group", kicker="Backup · Diagnosis · val")
    add_picture(slide, FIGURES / "paper_agreement_val.png", MARGIN, Inches(1.55), CONTENT_W, Inches(3.5))
    share = r.num(r.val("agree_share"), 0)
    add_bullets(slide, [
        f"Cache agrees with CLIP ({share}% of val samples): accuracy {r.num(r.val('base_acc_agree'))} → "
        f"{r.num(r.val('tda_acc_agree'))}, gap {r.num(r.val('base_gap_agree'), 1, sign=True)} → {r.num(r.val('tda_gap_agree'), 1, sign=True)} points.",
        f"Cache disagrees: accuracy {r.num(r.val('base_acc_disagree'))} → {r.num(r.val('tda_acc_disagree'))}, gap "
        f"{r.num(r.val('base_gap_disagree'), 1, sign=True)} → {r.num(r.val('tda_gap_disagree'), 1, sign=True)}.",
        "The echo's size is unrelated to CLIP's own calibration, so it overshoots except where CLIP was underconfident.",
    ], top=Inches(5.15), spacing=6, size=SUB)
    set_notes(slide, f"""
The lemma says what can happen; the data say how much. On the val splits the cache agrees with CLIP
on {share} percent of the samples. There, the positive cache leaves accuracy essentially unchanged, but
the confidence minus accuracy gap jumps from {r.num(r.val('base_gap_agree'), 1, sign=True)} to {r.num(r.val('tda_gap_agree'), 1, sign=True)} points. Where it disagrees,
it carries the accuracy gain with a much smaller change in the gap. That points directly to
the method: keep the disagreeing branch, drop the agreeing one.
""")
    return slide


def slide_method(prs, r):
    slide = new_slide(prs, "The agreement gate: use the cache only where it disagrees", kicker="4 · Method")
    add_panel(slide, MARGIN, Inches(1.65), CONTENT_W, Inches(1.4), fill=TINT_ACCENT, line=ACCENT)
    add_equation(slide, r"o(x) = b(x) + \alpha(x)\,c(x), \qquad \alpha(x) = \alpha\cdot\mathbf{1}\left[\arg\max_k c_k(x) \neq \hat y_0(x)\right]",
                 MARGIN, Inches(1.95), size=28, center_width=CONTENT_W)
    add_bullets(slide, [
        "Caches, admission and the negative cache are TDA's, unchanged.",
        "No parameter, no label, one argmax comparison per image; always on.",
        "Removes the echo at its source instead of calibrating it afterwards.",
        "Fixed on the val splits before any test, development or fresh result was computed.",
        "An inference-time instance of “update only on disagreement” (Malach & Shalev-Shwartz, NeurIPS'17).",
    ], top=Inches(3.35), spacing=12)
    set_notes(slide, """
The method follows directly. We keep everything in TDA, and only switch the positive cache off on
samples where its top class agrees with CLIP's pseudo-label. There it could only inflate confidence.
Where it disagrees, the cache is used exactly as in TDA. The rule has no parameter and needs no label;
it costs one comparison per image. It was fixed on the val splits before we looked at any test,
development or fresh results.
""")
    return slide


# --- Results ---------------------------------------------------------------------------------
def slide_main(prs, r):
    slide = new_slide(prs, "Main result: TDA's accuracy, half its ECE", kicker="5 · Results · test")
    rows = []
    for ds in CORE + ("mean",):
        rows.append([NAMES.get(ds, "Mean")] + [r.num(r.main(m, metric, ds)) for metric in ("accuracy", "ece")
                                               for m in ("clipzs", "tda", "cal_tda_agree")])
    add_table(slide, ["dataset", "CLIP", "TDA", "gate", "CLIP", "TDA", "gate"], rows, MARGIN, Inches(1.75),
              Inches(8.0), col_widths=[1.8, 1, 1, 1, 1, 1, 1], row_height=Inches(0.46), font_size=17, highlight_rows=(5,))
    add_text(slide, MARGIN, Inches(1.35), Inches(8.0), Inches(0.35),
             [("                                   accuracy (%)                                      ECE (%)", SMALL, True, MUTED)])
    pa, pe = r.pair("cal_tda_agree", "accuracy"), r.pair("cal_tda_agree", "ece")
    n = r.num(pe.n, 0)
    add_text(slide, Inches(9.0), Inches(1.8), Inches(3.7), Inches(4.2), [
        (f"Gate − TDA, {n} seeds (t-interval)", SUB, True, ACCENT),
        (f"accuracy {r.num(pa.mean_diff, sign=True)}  [{r.num(pa.ci90_lo, sign=True)}, {r.num(pa.ci90_hi, sign=True)}]", SUB, False, INK),
        (f"ECE {r.num(pe.mean_diff, sign=True)}  [{r.num(pe.ci90_lo, sign=True)}, {r.num(pe.ci90_hi, sign=True)}]", SUB, True, INK),
        (f"ECE lower on {r.num(pe.wins, 0)} of {n} seeds", SUB, False, INK),
        ("Worse only on Pets (CLIP underconfident).", SUB, False, MUTED),
    ], spacing=10)
    set_notes(slide, f"""
The main result on the five test sets, {n} stream orders. The gate keeps TDA's accuracy, a difference
of {r.num(pa.mean_diff, sign=True)} points, and lowers ECE by {r.num(-pe.mean_diff)} points, on every seed. Its mean ECE,
{r.num(r.main('cal_tda_agree', 'ece'))}, is below CLIP's {r.num(r.main('clipzs', 'ece'))}. It is far better than TDA on DTD, Flowers, EuroSAT and Aircraft, and
worse on Pets, where, as we saw, CLIP is underconfident and the echo happened to help.
""")
    return slide


def slide_reliability(prs, r):
    slide = new_slide(prs, "Backup: reliability, the gate stays near the diagonal", kicker="Backup · Results · test")
    add_picture(slide, FIGURES / "story_reliability_test.png", MARGIN, Inches(1.7), CONTENT_W, Inches(3.4))
    add_bullets(slide, [
        "Below the diagonal = overconfident. TDA (red) sits below it in most confident bins.",
        "The gate (green) tracks CLIP's curve where the cache agreed, and TDA's accuracy gain where it disagreed.",
    ], top=Inches(5.3), spacing=8)
    set_notes(slide, """
Reliability diagrams on the five test sets, all seeds pooled, ten bins. Points below the diagonal
are overconfident. TDA, in red, sits below the diagonal in the confident bins on most datasets. The
gate, in green, follows CLIP's calibration where the cache agreed with CLIP and keeps TDA's corrected
decisions where it disagreed. Pets is the one panel where TDA's curve is closer to the diagonal.
""")
    return slide


COMPARISON = (("TDA", "TDA (reference)"), ("agreement gate", "agreement gate (ours)"),
              ("read-first TDA", "read-before-write TDA"), ("TDA + T (val)", "TDA + temperature (val, labels)"),
              ("TDA + ZAEC", "TDA + ZAEC (post-hoc)"), ("gate + ZAEC", "gate + ZAEC"),
              ("DMN-weak", "DMN-weak"), ("ECALP", "ECALP"))


def slide_comparison(prs, r):
    slide = new_slide(prs, "Against calibration fixes and other memory methods", kicker="5 · Results · test")
    rows = []
    for method, label in COMPARISON:
        try:
            acc, ece = r.comp(method, "accuracy"), r.comp(method, "ece")
        except KeyError:
            rows.append([label, "running", "", ""])
            continue
        delta = "—" if method == "TDA" else r.num(r.comp_pair(method, "ece").mean_diff, sign=True)
        rows.append([label, r.num(acc), r.num(ece), delta])
    add_table(slide, ["method (mean of 5, 3 seeds)", "acc.", "ECE", "ΔECE vs TDA"], rows, MARGIN, Inches(1.7),
              Inches(7.3), col_widths=[3.4, 1, 1, 1.5], row_height=Inches(0.47), font_size=16, highlight_rows=(1,))
    add_picture(slide, FIGURES / "paper_plane_test.png", Inches(8.1), Inches(1.6), Inches(4.7), Inches(4.0))
    add_text(slide, MARGIN, Inches(6.05), Inches(12.0), Inches(0.75), [
        ("Removing only the self-vote removes part of the excess ECE. Post-hoc temperatures (fitted on val, or "
         "ZAEC, label-free) help less than the gate; ZAEC on top of the gate helps here, not on average on dev and "
         "fresh.", SUB, False, INK)])
    set_notes(slide, f"""
Here we compare with the closest alternatives, three seeds, all paired to TDA. Removing only the
self-vote, by reading before writing, reaches {r.num(r.comp('read-first TDA', 'ece'))}. A single temperature fitted on val with labels
reaches {r.num(r.comp('TDA + T (val)', 'ece'))}. ZAEC, a recent label-free post-hoc method that restores the zero-shot entropy, reaches
{r.num(r.comp('TDA + ZAEC', 'ece'))}. The gate reaches {r.num(r.comp('agreement gate', 'ece'))} without any calibration step, because it removes the cause instead
of rescaling afterwards. Gate plus ZAEC reaches {r.num(r.comp('gate + ZAEC', 'ece'))} here, but that stacking does not hold up on
average on the development and fresh sets, so we do not claim it. DMN-weak and ECALP have higher ECE than TDA; ECALP is the
most accurate method here, at {r.num(r.comp('ECALP', 'accuracy'))} percent, but has an ECE of {r.num(r.comp('ECALP', 'ece'))}.
""")
    return slide


ABL_TIERS = ("val", "test", "dev", "fresh")
ABL_ROWS = (("clip", "CLIP"), ("base", "+ negative cache = b"), ("apos", "b + positive cache, agreeing only"),
            ("gate", "b + positive cache, disagreeing only (the gate)"), ("tda", "b + positive cache, all (TDA)"))


def slide_ablation_groups(prs, r):
    slide = new_slide(prs, "Ablation: the agreeing positive cache adds only confidence", kicker="4 · Diagnosis · all tiers")
    a = r.d["ni_abl"].set_index("tier")
    rows = [[label] + [f"{r.num(a.loc[t, f'{k}_acc'], 1)} / {r.num(a.loc[t, f'{k}_ece'], 1)}" for t in ABL_TIERS]
            for k, label in ABL_ROWS]
    add_table(slide, ["accuracy / ECE (P2, 3 seeds)"] + list(ABL_TIERS), rows, MARGIN, Inches(1.6), CONTENT_W,
              col_widths=[5.0, 1.8, 1.8, 1.8, 1.8], row_height=Inches(0.46), font_size=16, highlight_rows=(3,))
    d_apos = (a.apos_acc - a.base_acc).abs().max()
    add_bullets(slide, [
        f"Agreeing samples ({r.num(a.loc['val', 'agree_share'], 0)}% of val): accuracy moves by at most "
        f"{r.num(d_apos)} points on any tier, while ECE roughly doubles on val, test and dev. No information, only "
        "confidence (Lemma 2).",
        "Disagreeing samples carry TDA's whole accuracy gain; the gate keeps it at about half TDA's ECE.",
        "Fresh: the agreeing echo even lowers ECE, because CLIP is underconfident there.",
    ], top=Inches(4.65), spacing=8, size=SUB)
    set_notes(slide, f"""
The ablation that tests the diagnosis on every tier. We add TDA's positive cache to CLIP plus the
negative cache only on the samples where it agrees with CLIP, only where it disagrees, which is the
gate, or everywhere, which is TDA. On the agreeing samples it changes accuracy by at most {r.num(d_apos)} points,
on every tier, but the expected calibration error roughly doubles on val, test and the development sets,
from {r.num(a.loc['val', 'base_ece'], 1)} to {r.num(a.loc['val', 'apos_ece'], 1)} on val. So on the agreeing group the positive cache carries no information, only
confidence, exactly as the lemma says. The whole accuracy gain comes from the disagreeing samples, and
the gate keeps it. On the fresh tier the agreeing echo even helps calibration, because CLIP is
underconfident there; that is the limitation we come back to.
""")
    return slide


def slide_no_info(prs, r):
    slide = new_slide(prs, "Beyond one bit: CLIP's and the cache's scores say almost nothing",
                      kicker="5 · Results · all tiers")
    add_picture(slide, FIGURES / "no_info_auc.png", MARGIN, Inches(1.5), Inches(7.2), Inches(4.6))
    auc, flt = r.d["ni_auc"], r.d["ni_filter"].set_index("tier")
    comb = auc[auc.measure == "combined"].set_index("tier").auc
    held = auc[(auc.measure != "combined") & auc.tier.isin(("dev", "fresh"))].auc
    add_bullets(slide, [
        "Flip: TDA's decision differs from CLIP's. Can label-free scores tell right flips from wrong ones?",
        f"Ten scores of CLIP, of the cache and of their relation: AUC {r.num(held.min())}–{r.num(held.max())} on dev "
        "and fresh; the ones that look useful on val/test reverse on fresh.",
        f"All ten combined (fitted on val): AUC {r.num(comb['test'])} on test, {r.num(comb['dev'])} on dev, "
        f"{r.num(comb['fresh'])} on fresh. It learns datasets, not flips.",
        f"As a flip filter: at most {r.num(flt.filter_gain.abs().max())} accuracy points; an oracle would get "
        f"{r.num(flt.oracle_pick.min(), 1)}–{r.num(flt.oracle_pick.max(), 1)}.",
        "Agreement is useful as one bit; finer rules cannot beat the gate.",
    ], left=Inches(7.9), top=Inches(1.6), width=Inches(4.9), spacing=8, size=SUB)
    set_notes(slide, f"""
Could a finer use of the scores beat the one-bit gate? TDA's accuracy gain comes from flips, samples where
its decision differs from CLIP's. We took ten label-free quantities: CLIP's confidence, margin and the
probability it gives the cache's class; the cache's confidence, mass and weight on CLIP's class; and
their relation, the correlation over classes, the JS divergence, the weight gap and the alpha needed to
flip. On the development and fresh sets every one of them separates right from wrong flips with an AUC
between {r.num(held.min())} and {r.num(held.max())}, close to chance. All ten combined in a logistic regression fitted on val reach
{r.num(comb['test'])} on test, which has the same datasets as val, but {r.num(comb['dev'])} and {r.num(comb['fresh'])} on new datasets: they learn which
datasets flip well, not which flips are right. Used as a filter, that changes accuracy by at most {r.num(flt.filter_gain.abs().max())}
points, while an oracle picking the right one of CLIP's and the cache's class would add {r.num(flt.oracle_pick.min(), 1)} to
{r.num(flt.oracle_pick.max(), 1)}. The headroom exists, but this information is not in the scores.
""")
    return slide


def slide_generalization(prs, r):
    slide = new_slide(prs, "Generalization: datasets never used in the design", kicker="5 · Results · dev + fresh")
    g = r.gen
    rows = []
    for _, x in g.iterrows():
        rows.append([("dev  " if x.tier == "dev" else "fresh  ") + NAMES.get(x.dataset, x.dataset),
                     r.num(x.tda_accuracy), r.num(x.gate_accuracy), r.num(x.tda_ece), r.num(x.gate_ece),
                     r.num(x.d_ece, sign=True)])
    add_table(slide, ["dataset", "TDA acc.", "gate acc.", "TDA ECE", "gate ECE", "ΔECE"], rows, MARGIN, Inches(1.55),
              Inches(8.2), col_widths=[2.6, 1, 1, 1, 1, 1], row_height=Inches(0.33), font_size=14)
    wins, n = int((g.d_ece < 0).sum()), len(g)
    add_text(slide, Inches(9.1), Inches(1.7), Inches(3.6), Inches(4.5), [
        (f"{r.num(n, 0)} datasets", SUB, True, ACCENT),
        (f"ECE lower on {r.num(wins, 0)}", SUB, False, INK),
        (f"mean ΔECE {r.num(g.d_ece.mean(), sign=True)}", SUB, True, INK),
        (f"mean Δaccuracy {r.num(g.d_accuracy.mean(), sign=True)}", SUB, False, INK),
        (f"worst Δaccuracy {r.num(g.d_accuracy.min(), sign=True)}", SUB, False, MUTED),
        ("P2, 3 seeds; nothing fitted on these sets.", SMALL, False, MUTED),
    ], spacing=10)
    set_notes(slide, f"""
The method was designed on the val splits of five datasets. Here are {r.num(n, 0)} datasets that played no
role in the design: four development sets and the fresh sets, including the full ImageNet validation
set. The gate lowers ECE on {r.num(wins, 0)} of them; the mean ECE change is {r.num(g.d_ece.mean(), sign=True)} points and the
mean accuracy change {r.num(g.d_accuracy.mean(), sign=True)} points. Accuracy is kept everywhere; ECE is mixed, and the next
slide shows when it helps.
""")
    return slide


def slide_where(prs, r):
    slide = new_slide(prs, "Where it helps: mostly where CLIP is not underconfident", kicker="5 · Results · all tiers")
    rg, rc = r.rho("clip_gap"), r.rho("conf")
    add_picture(slide, FIGURES / "paper_where_all.png", MARGIN, Inches(1.6), Inches(6.6), Inches(4.9))
    add_bullets(slide, [
        "Prediction from the lemmas: on agreeing samples the echo only adds confidence. Removing it helps if CLIP "
        "is calibrated or overconfident, and hurts if CLIP is underconfident.",
        f"Spearman ρ of ΔECE with CLIP's own confidence − accuracy gap: {r.num(rg.rho, sign=True)} "
        f"(n = {r.num(rg.n, 0)}). A tendency, not a rule: GTSRB and Food101 lose a little, VOC2007 and Cars gain. "
        "Uses labels: it explains; a label-free switch is exploratory (backup).",
        f"Label-free proxy (CLIP's mean confidence on the first {r.num(SIGNAL_HEAD, 0)} samples): ρ = "
        f"{r.num(rc.rho, sign=True)}. Confidence is not calibration: CIFAR-100 is unconfident and underconfident.",
        "The gate stays always on; this is its stated condition.",
    ], left=Inches(7.5), top=Inches(1.7), width=Inches(5.2), spacing=10, size=SUB)
    set_notes(slide, f"""
When does the gate help? The lemmas say that on agreeing samples the echo only adds confidence. So
removing it should help when CLIP is calibrated or overconfident, and hurt when CLIP is underconfident.
That is the tendency we see: over {r.num(rg.n, 0)} datasets, the gate's ECE change follows CLIP's own confidence minus
accuracy gap with a rank correlation of {r.num(rg.rho, sign=True)}. That variable needs labels, so it explains the
result but cannot be used to switch the method. The best label-free proxy we tried, CLIP's mean confidence,
reaches only {r.num(rc.rho, sign=True)}, because a dataset can be unconfident and still underconfident.
""")
    return slide


def slide_conclusion(prs, r):
    slide = new_slide(prs, "Limitations and conclusion", kicker="5 · Conclusion")
    add_bullets(slide, [
        "Limitations",
        (1, "Underconfident CLIP (Pets, Caltech101, CIFAR-10/100, ImageNet-R, STL-10): the echo happens to help, the gate is worse; "
            "a confidence switch recovers the confident ones (exploratory, backup)."),
        (1, "The disagreeing branch is still overconfident on hard sets; ZAEC on top helps on test, not on average elsewhere."),
        (1, "One backbone, one cache method. The argument applies to any cache of the model's own pseudo-labels (e.g. DMN), untested."),
        (1, "Designed on val after earlier attempts, some explored on test; dev and fresh tiers are the clean evidence."),
        "Conclusion",
        (1, "TDA's positive cache echoes CLIP: where it agrees it cannot change a decision, only inflate confidence."),
        (1, "The agreement gate keeps TDA's accuracy everywhere and halves its ECE on the test and development sets, "
            "with no parameter and no label; on the fresh tier the gain is smaller and not uniform."),
        (1, "Agreement is one bit: the agreeing positive cache adds no accuracy, and CLIP's and the cache's scores say "
            "almost nothing about which other flips are right, so finer rules cannot beat the gate."),
    ], spacing=8)
    set_notes(slide, """
To close. The gate is worse on datasets where CLIP is underconfident, because there the echo happened
to correct CLIP. A label-free confidence switch recovers the confident ones, but we found it after the
fresh results, so it stays an exploratory extension in the backup. The remaining overconfidence is in
the disagreeing branch. We tested one backbone and one cache method, and the design was done on val
after earlier exploration on test, which is why the development and fresh sets matter. We also tried
finer uses of agreement, an alpha graded by disagreement or by the correlation between cache and CLIP,
and flip rules; none held up on held-out data. The main message: TDA is overconfident largely because
its cache echoes CLIP, and gating it on disagreement removes the echo at unchanged accuracy.
""")
    return slide


# --- Backup ---------------------------------------------------------------------------------
def slide_b_p1(prs, r):
    slide = new_slide(prs, "Backup: protocol P1 (published per-dataset α, β)", kicker="Backup")
    p = r.d["p1"][r.d["p1"].dataset == "mean"]
    rows = []
    for m, label in (("clipzs", "CLIP"), ("tda", "TDA"), ("cal_tda_agree", "agreement gate")):
        a = p[(p.method == m) & (p.metric == "accuracy")].iloc[0]
        e = p[(p.method == m) & (p.metric == "ece")].iloc[0]
        rows.append([label, r.num(a.mean_P1), r.num(a.mean_P2), r.num(e.mean_P1), r.num(e.mean_P2)])
    add_table(slide, ["method (test, mean of 5, 3 seeds)", "acc. P1", "acc. P2", "ECE P1", "ECE P2"], rows,
              MARGIN, Inches(1.8), Inches(9.0), col_widths=[3.2, 1, 1, 1, 1], row_height=Inches(0.5), font_size=18)
    repro = r.d["repro"].delta_gpu_vs_benchmark.abs().max()
    add_bullets(slide, [
        f"Our GPU run reproduces the benchmark's published CLIP and TDA accuracies within {r.num(repro)} points per dataset.",
        "P1 tunes α, β per dataset (on test, in the original paper); P2 is one setting for all.",
    ], top=Inches(4.2), spacing=10)
    set_notes(slide, f"""
For completeness, the published protocol P1 with TDA's per-dataset alpha and beta. The conclusions
do not change: the gate keeps TDA's accuracy and lowers its ECE under both protocols. Our GPU
reproduction of the benchmark is within {r.num(repro)} points per dataset, and every number in this talk
comes from an exact CPU replay of cached features.
""")
    return slide


def slide_b_proofs(prs, r):
    slide = new_slide(prs, "Backup: proofs", kicker="Backup")
    add_equation(slide, r"b_{\hat y_0} - b_j \geq (z_{\hat y_0} - z_j) - (\max n - \min n) > \alpha S"
                 r"\;\Rightarrow\; o_{\hat y_0} - o_j > \alpha S - \alpha S = 0",
                 MARGIN, Inches(1.8), size=24, center_width=CONTENT_W)
    add_equation(slide, r"\max\,\mathrm{softmax}(o) = \left(1 + \sum_{j \neq \hat y} e^{-(o_{\hat y} - o_j)}\right)^{-1}",
                 MARGIN, Inches(2.9), size=24, center_width=CONTENT_W)
    add_bullets(slide, [
        "No-flip region (first line): uses 0 ≤ c_k ≤ S.",
        "Agreement lemma: every margin of o is at least the margin of b, and the top probability is increasing "
        "in every margin (second line).",
        "Self-vote: an admitted x has c_ŷ₀(x) ≥ exp(0) = 1, and every other vote is at most 1.",
    ], top=Inches(4.0), spacing=10)
    set_notes(slide, """
The two proofs in full. For the no-flip region, the negative cache can move a margin by at most its
range, and the positive cache by at most alpha times the shot capacity, so a larger CLIP gap cannot be
overturned. For the agreement lemma, each output margin is at least the corresponding margin of CLIP
plus negative cache, and the top softmax probability is an increasing function of the margins.
""")
    return slide


def slide_b_earlier(prs, r):
    slide = new_slide(prs, "Backup: earlier attempts", kicker="Backup")
    rows = []
    for m, label in (("cal_tda_agree", "agreement gate (final)"),) + tuple(ra.EARLIER.items()):
        rows.append([label, r.num(r.main(m, "accuracy")), r.num(r.main(m, "ece")),
                     r.num(r.pair(m, "accuracy").mean_diff, sign=True), r.num(r.pair(m, "ece").mean_diff, sign=True)])
    add_table(slide, ["variant (test, P2, mean of 5)", "acc.", "ECE", "Δacc. vs TDA", "ΔECE vs TDA"], rows,
              MARGIN, Inches(1.8), Inches(10.5), col_widths=[3.6, 1, 1, 1.4, 1.4], row_height=Inches(0.48),
              font_size=17, highlight_rows=(0,))
    add_text(slide, MARGIN, Inches(5.4), CONTENT_W, Inches(1.0), [
        ("Margin admission starves the cache; fusion caps confidence; the two-way-tie gate calibrates but loses "
         "accuracy; the LOO temperature calibrates partly. Explored before the diagnosis, partly on test.", SUB, False, INK)])
    set_notes(slide, """
These are the changes we tried before the diagnosis, all on the same protocol and paired to TDA.
Margin admission starves the cache on hard datasets. Probability fusion caps every prediction's
confidence. Gating fusion to two-way ties fixed calibration but lost accuracy. A leave-one-out
temperature helped calibration partly. Some of this exploration was done on test, which we disclose.
""")
    return slide


def slide_b_per_dataset(prs, r):
    slide = new_slide(prs, "Backup: comparison per dataset (ECE, test, 3 seeds)", kicker="Backup")
    rows = []
    for method, label in COMPARISON:
        cells = []
        for ds in CORE:
            try:
                cells.append(r.num(r.comp(method, "ece", ds)))
            except KeyError:
                cells.append("—")
        rows.append([label] + cells)
    add_table(slide, ["method"] + [NAMES[d] for d in CORE], rows, MARGIN, Inches(1.7), CONTENT_W,
              col_widths=[3.4, 1, 1, 1, 1, 1], row_height=Inches(0.47), font_size=16, highlight_rows=(1,))
    set_notes(slide, """
Per dataset ECE for every method in the comparison table, three seeds each. The gate is best on
EuroSAT; the val-fitted temperature wins on DTD and Aircraft, ZAEC on Flowers, and plain TDA on Pets,
where removing the echo costs calibration. Combining the gate with ZAEC is the strongest on the mean.
""")
    return slide


def slide_b_switch(prs, r):
    slide = new_slide(prs, "Backup (exploratory): keep the echo on confident streams", kicker="Backup")
    sw = r.d["switch"]
    mean = sw[sw.dataset == "mean"]
    tiers = (("val", "val*"), ("dev", "dev*"), ("test", "test"), ("fresh", "fresh"))
    rows = []
    for method, _ in ra.SWITCH_METHODS:
        cells = []
        for tier, _ in tiers:
            row = mean[(mean.tier == tier) & (mean.method == method)].iloc[0]
            cells.append(r.num(row.ece))
        rows.append([method] + cells)
    add_table(slide, ["mean ECE (P2, 3 seeds)"] + [label for _, label in tiers], rows, MARGIN, Inches(1.7), Inches(10.5),
              col_widths=[3.6, 1.2, 1.2, 1.2, 1.2], row_height=Inches(0.5), font_size=17, highlight_rows=(3,))
    thresh = float(sw.threshold.iloc[0])
    per = sw[(sw.dataset != "mean") & (sw.method == "confidence switch") & (sw.tier != "val")]
    on = per[per.switch_on_pct > 50].dataset.unique()
    add_text(slide, MARGIN, Inches(4.6), CONTENT_W, Inches(1.8), [
        (f"Rule: keep TDA's logits while the running mean CLIP confidence is above {r.num(thresh)}; gate otherwise. "
         "Online, label-free, one threshold chosen on the design tiers (*: val + dev, 3 confident and 6 unconfident "
         "sets); test and fresh held out.", SUB, False, INK),
        ("Switches on for " + ", ".join(NAMES.get(x, x) for x in on) + "; they carry the whole gain. "
         "Found after the fresh results were known: not clean evidence.", SUB, False, MUTED)], spacing=8)
    fr = r.d["switch_sweep"][r.d["switch_sweep"].tier == "fresh"]
    best = fr[fr.ece <= fr.ece.min() + 0.1].threshold
    set_notes(slide, f"""
An exploratory extension, not the method. The gate loses where CLIP is underconfident, and on the
design sets CLIP is underconfident exactly on the confident ones. Keeping TDA's echo while the running
mean CLIP confidence stays above {r.num(thresh)} lowers mean ECE on the held-out test and fresh tiers with the
same accuracy. The threshold was chosen on val plus the development sets, which contain both kinds of
stream; any value in the flat region gives the same design result, and the fresh sets alone would have
preferred a lower one, {r.num(best.min())} to {r.num(best.max())}, so it is not tuned to them. Still, the gain comes from a handful
of datasets, CIFAR-100 and ImageNet-R are missed, and we found the idea after seeing the fresh results,
so it needs new datasets before it can be claimed.
""")
    return slide


def slide_b_rates(prs, r):
    slide = new_slide(prs, "Backup: share of right flips along each label-free score", kicker="Backup · all tiers")
    add_picture(slide, FIGURES / "no_info_rates.png", MARGIN, Inches(1.5), CONTENT_W, Inches(4.4))
    rates = r.d["ni_rates"]
    low = rates[(rates.measure == "alpha_needed") & (rates.n >= 30)].right_pct.min()
    add_callout(slide, "Slopes seen on val/test flatten or reverse on fresh. The α needed to flip keeps its direction, "
                f"but even its worst group of flips is right {r.num(low, 1)}% of the time: nothing to block.",
                top=Inches(6.0), height=Inches(0.75))
    set_notes(slide, f"""
The same question as a rate. For each of the ten label-free scores, the flips are split into eight equal
groups, and we plot the share of them that are right, one line per tier. Where a line slopes on val and
test, it is flat or reverses on the fresh tier. The one score whose direction holds everywhere is the
alpha needed to flip: harder flips are right less often. But even the worst group is right {r.num(low, 1)} percent
of the time, so blocking flips by that score cannot raise accuracy.
""")
    return slide


def slide_b_negative(prs, r):
    slide = new_slide(prs, "Backup: finer uses of agreement that did not hold up", kicker="Backup")
    add_bullets(slide, [
        "Chosen on val, checked on test, development and fresh sets:",
        (1, "α graded by how strongly cache and CLIP disagree: reduces to a rescaled constant α."),
        (1, "α from the correlation between cache and CLIP logits (linear, inverse, bump at mid correlation): "
            "the val gain reverses on test and fresh."),
        (1, "Softening the disagreeing branch by the cache–CLIP conflict: helps on test, hurts on dev and fresh."),
        (1, "Agreement rate as a label-free accuracy estimate: worse than CLIP's own confidence (the cache votes with CLIP's labels)."),
        (1, "Flip rules on disagreeing samples: picking the right of CLIP and cache per sample would add several points, "
            "but no label-free rule captures it."),
        "Agreement is useful as a binary gate; nothing finer generalized.",
    ], spacing=10)
    set_notes(slide, """
We also tried to get more out of the agreement signal than a binary gate. Making alpha depend on how
strongly the cache disagrees with CLIP, or on the correlation between the cache and CLIP logits, including
a bump that is strongest at intermediate correlation, gave gains on val that reversed on the test and fresh
sets. Softening the disagreeing branch helped on test only. The agreement rate is a worse accuracy
estimate than CLIP's own confidence, because the cache is filled with CLIP's own pseudo-labels. And while an
oracle that picks the right one of CLIP and the cache per sample would add several points of accuracy, no
label-free flip rule captured any of it. So the binary gate is what agreement gives us.
""")
    return slide


MAIN_BUILDERS = [
    slide_title, slide_agenda,
    slide_task, slide_clip,
    slide_tda_why, slide_tda_pipeline, slide_tda_math,
    slide_calibration, slide_tda_overconfident, slide_tda_caches,
    slide_selfvote, slide_lemmas, slide_ablation_groups,
    slide_method,
    slide_main, slide_comparison, slide_generalization, slide_where, slide_no_info,
    slide_conclusion,
]
BACKUP_BUILDERS = [slide_agreement, slide_reliability, slide_b_p1, slide_b_proofs, slide_b_earlier, slide_b_per_dataset, slide_b_switch,
                   slide_b_rates, slide_b_negative]
SLIDE_BUILDERS = MAIN_BUILDERS + BACKUP_BUILDERS
TDA_SECTION = [slide_tda_why, slide_tda_pipeline, slide_tda_math, slide_tda_overconfident, slide_tda_caches]


def slide_heading(slide, index):
    if index == 1:
        return "Title"
    for shape in slide.shapes:
        if shape.name == "title":
            return shape.text_frame.text.strip()
    return f"Slide {index}"


def write_markdown_mirror(prs, md_path) -> Path:
    lines = [
        "# The cache that echoes CLIP: agreement-gated TDA — deck v5",
        "",
        "**Generated file — do not edit.** Produced by `python presentation/build_pptx.py`",
        "alongside `calibration_aware_tta.pptx`; edit the slide functions in that script instead.",
        "",
        "Each section is one slide: its on-screen heading and the speaker notes from its notes pane.",
        "",
    ]
    for index, slide in enumerate(prs.slides, start=1):
        notes = slide.notes_slide.notes_text_frame.text.strip() if slide.has_notes_slide else ""
        lines += ["---", "", f"## {index}. {slide_heading(slide, index)}", ""]
        lines += [notes, ""] if notes else []
    md_path = Path(md_path)
    md_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return md_path


def build(out_path, markdown_path=None, root=ROOT):
    """Build the deck; returns (path, results) so tests can inspect what was emitted."""
    results = Results(root)
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    n_main = len(MAIN_BUILDERS)
    for index, builder in enumerate(SLIDE_BUILDERS, start=1):
        slide = builder(prs, results)
        if index > 1:
            label = None if index <= n_main else f"B{index - n_main}"
            add_footer(slide, index, n_main, label)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out_path))
    if markdown_path:
        write_markdown_mirror(prs, markdown_path)
    return out_path, results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="presentation/calibration_aware_tta.pptx")
    parser.add_argument("--markdown", default="presentation/slides.md")
    args = parser.parse_args()
    path, _ = build(args.out, markdown_path=args.markdown or None)
    print(f"wrote {path} ({len(SLIDE_BUILDERS)} slides: {len(MAIN_BUILDERS)} main + {len(BACKUP_BUILDERS)} backup)")


if __name__ == "__main__":
    main()
