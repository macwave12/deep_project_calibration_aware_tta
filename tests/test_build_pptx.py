"""The deck is built from tracked CSVs and records; these tests pin that."""

from pathlib import Path

import pandas as pd
import pytest
from pptx import Presentation
from pptx.util import Pt

from presentation.build_pptx import (
    BACKUP_BUILDERS, MAIN_BUILDERS, SLIDE_BUILDERS, TDA_SECTION, build, number_tokens,
)

ROOT = Path(__file__).resolve().parents[1]


def _slide_text(slide, *, skip_footer=True):
    parts = []
    for shape in slide.shapes:
        if skip_footer and shape.name == "footer":
            continue
        if shape.has_text_frame:
            parts.append(shape.text_frame.text)
        if shape.has_table:
            for row in shape.table.rows:
                parts.extend(cell.text for cell in row.cells)
    return "\n".join(parts)


@pytest.fixture(scope="module")
def deck(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("deck")
    out, results = build(out_dir / "deck.pptx", markdown_path=out_dir / "slides.md")
    prs = Presentation(str(out))
    texts = [_slide_text(s) for s in prs.slides]
    notes = [s.notes_slide.notes_text_frame.text for s in prs.slides]
    return prs, texts, notes, (out_dir / "slides.md").read_text(encoding="utf-8"), results


def test_slide_count_and_every_slide_has_speaker_notes(deck):
    prs, _, notes, markdown, _ = deck
    assert len(prs.slides) == len(SLIDE_BUILDERS)
    assert len(MAIN_BUILDERS) <= 20  # T16 plan §10: at most 20 main slides, backup after
    assert len(BACKUP_BUILDERS) >= 3
    for index, note in enumerate(notes, start=1):
        assert len(note.split()) >= 25, f"slide {index} has no real speaker notes"
    assert markdown.count("\n## ") == len(SLIDE_BUILDERS)
    for note in notes:   # the markdown mirrors the notes verbatim
        assert note.strip() in markdown


def test_number_scanner_finds_data_but_not_names():
    text = "ECE 17.41 and 1,692, rose −5.52 (p > 0.03) on ViT-B/16, CVPR'24, A3, top-2, 16-shot, 100 classes"
    assert number_tokens(text) == ["17.41", "1,692", "5.52", "0.03", "100"]


def test_tda_section_is_at_least_a_quarter_of_the_main_deck():
    assert len(TDA_SECTION) / len(MAIN_BUILDERS) >= 0.25


def test_slides_name_no_code_identifiers(deck):
    _, texts, _, _, _ = deck
    for index, text in enumerate(texts, start=1):
        for token in ("cal_tda", "clipzs", "fusion_gate", "gate_temp"):
            assert token not in text, (index, token)


def test_every_number_on_a_slide_or_in_the_notes_came_from_data(deck):
    """Each numeral must have passed through Results.num (data) or Results.illus.

    Results.num only formats values read from tracked CSVs/records; a number
    typed into a slide string would not be in either set and fails here.
    """
    _, texts, notes, _, results = deck
    allowed = results.emitted | results.illustrative
    for index, text in enumerate(texts + notes, start=1):
        for token in number_tokens(text):
            if len(token) == 1:      # single digits: counts, indices, "exp(0) = 1"
                continue
            assert token in allowed, f"slide/notes {index}: '{token}' was not produced from data"


def test_illustrative_numbers_are_labelled_on_their_slide(deck):
    _, texts, _, _, results = deck
    only_illustrative = results.illustrative - results.emitted
    for index, text in enumerate(texts, start=1):
        tokens = {t for t in number_tokens(text) if len(t) > 1}
        if tokens & only_illustrative:
            assert "llustrative" in text, f"slide {index} shows illustrative numbers without saying so"


def test_fonts_are_projector_legible(deck):
    prs = deck[0]
    for index, slide in enumerate(prs.slides, start=1):
        for shape in slide.shapes:
            if shape.name == "footer":
                continue
            frames = []
            if shape.has_text_frame:
                frames.append(shape.text_frame)
            if shape.has_table:
                frames.extend(cell.text_frame for row in shape.table.rows for cell in row.cells)
            for frame in frames:
                for paragraph in frame.paragraphs:
                    for run in paragraph.runs:
                        if run.text.strip() and run.font.size is not None:
                            assert run.font.size >= Pt(14), (index, run.text[:40])


def test_main_result_slide_quotes_the_ten_seed_records(deck):
    _, texts, _, _, _ = deck
    prot = pd.read_csv(ROOT / "results/protocols_10seeds.csv")
    slide = next(t for t in texts if "Main result" in t)
    for method in ("clipzs", "tda", "cal_tda_agree"):
        for metric in ("accuracy", "ece"):
            row = prot[(prot.protocol == "P2") & (prot.dataset == "mean") & (prot.method == method) & (prot.metric == metric)]
            assert f"{float(row.iloc[0]['mean']):.2f}" in slide, (method, metric)


def test_tda_section_shows_and_cites_the_tda_paper_figures(deck):
    prs, texts, _, _, _ = deck
    cited = [t for t in texts if "arxiv.org/abs/2403.18293" in t]
    assert any("Fig. 1" in t for t in cited) and any("Fig. 2" in t for t in cited)


def test_every_evidence_slide_names_its_split(deck):
    _, texts, _, _, _ = deck
    assert sum("DIAGNOSIS · VAL" in t for t in texts) >= 2
    assert sum("RESULTS · TEST" in t for t in texts) >= 3
    assert sum("ALL TIERS" in t or "HELD-OUT" in t for t in texts) >= 4
    for t in texts:
        if "DIAGNOSIS · VAL" in t:
            assert "(test" not in t
