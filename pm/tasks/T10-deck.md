# T10 — Rebuild the deck

Status: open · Owner: deck agent · Depends on: T05–T09

Rewrite `presentation/build_pptx.py` to the outline in `PLAN.md`, pulling every number from
CSVs, with speaker notes in `slides.md`. Address every point in Idan's review of
2026-11-07 (listed in `pm/research/deck-review-2026-11-07.md`). Render to PDF/PNG for
review. Keep `tests/test_build_pptx.py` green and extend it to the new slides.
