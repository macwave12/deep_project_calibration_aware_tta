"""Regenerate notebooks/extract_features_colab.ipynb from its .py mirror.

The .py uses the "percent" cell format: a line ``# %%`` starts a code cell and
``# %% [markdown]`` a markdown cell whose lines are ``# ``-prefixed.

    python notebooks/sync_colab_notebook.py           # write the .ipynb
    python notebooks/sync_colab_notebook.py --check   # exit 1 if they differ

tests/test_colab_notebook.py runs the check.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = HERE / "extract_features_colab.py"
NB = HERE / "extract_features_colab.ipynb"

METADATA = {
    "accelerator": "GPU",
    "colab": {"name": "extract_features_colab.ipynb", "provenance": [], "gpuType": "T4"},
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
    "language_info": {"name": "python"},
}


def _strip_blank(lines):
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def parse_percent(text):
    """-> [(cell_type, source_text)]"""
    cells, kind, buf = [], None, []

    def flush():
        if kind is not None:
            lines = _strip_blank(list(buf))
            if kind == "markdown":
                lines = [l[2:] if l.startswith("# ") else l.lstrip("#") for l in lines]
            cells.append((kind, "\n".join(lines)))

    for line in text.splitlines():
        if line.startswith("# %%"):
            flush()
            kind = "markdown" if "[markdown]" in line else "code"
            buf = []
        elif kind is not None:
            buf.append(line)
    flush()
    return cells


def _source_lines(text):
    lines = text.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]] if text else []


def build_notebook(cells):
    out = []
    for kind, src in cells:
        cell = {"cell_type": kind, "metadata": {}, "source": _source_lines(src)}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        out.append(cell)
    return {"cells": out, "metadata": METADATA, "nbformat": 4, "nbformat_minor": 5}


def notebook_cells(nb):
    return [(c["cell_type"], "".join(c["source"])) for c in nb["cells"]]


def render(nb):
    return json.dumps(nb, indent=1, ensure_ascii=False) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--check", action="store_true")
    args = p.parse_args(argv)
    text = render(build_notebook(parse_percent(PY.read_text(encoding="utf-8"))))
    if args.check:
        same = NB.is_file() and NB.read_text(encoding="utf-8") == text
        print("in sync" if same else f"{NB.name} is out of date: run {Path(__file__).name}")
        return 0 if same else 1
    NB.write_text(text, encoding="utf-8")
    print(f"wrote {NB}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
