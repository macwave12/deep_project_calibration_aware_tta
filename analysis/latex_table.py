"""One place that decides how a generated table is laid out in the report.

`DataFrame.to_latex(caption=..., label=...)` emits the caption *above* the
tabular and never emits `\\centering`. The report wants the opposite
convention -- centred body, caption underneath -- so the generators here call
`to_latex()` for the tabular only and hand the result to `wrap_table`, which
builds the float itself. Keeping that in one function means the two table
generators cannot drift apart, and means re-running the analysis scripts
reproduces the report's layout instead of silently reverting it.
"""

from __future__ import annotations

import re


def _tidy_multiindex_header(body: str) -> str:
    """Fix the two things pandas does badly when the columns are a MultiIndex.

    1. Group spanners are emitted as `\\multicolumn{3}{r}{accuracy}`, so the word
       sits hard against the *last* column of its group -- on the page "ece"
       looks like a heading for the `tda` column alone. `c` centres it over the
       span it actually covers.
    2. When the index has a name, pandas emits an extra all-empty row carrying
       just that name (`dataset &  &  & ... \\\\`). It is pure noise between the
       header and the first data row.
    """
    body = re.sub(r"\\multicolumn\{(\d+)\}\{r\}", r"\\multicolumn{\1}{c}", body)
    body = re.sub(r"\n[A-Za-z_ ]+(?: &\s*)+\\\\(?=\n\\midrule)", "", body)
    return _rule_under_spanners(body)


def _rule_under_spanners(body: str) -> str:
    """Underline each group spanner with a `\\cmidrule`, so it is visibly attached
    to the columns it covers.

    Without one, a short spanner over right-aligned numeric columns sits almost
    exactly above the last sub-header in its group -- "ece" reads as a label for
    the `tda` column alone rather than for all three beneath it.
    """
    lines = body.split("\n")
    for i, line in enumerate(lines):
        if "\\multicolumn" not in line:
            continue
        spans, column = [], 1
        for cell in line.replace("\\\\", "").split("&"):
            match = re.search(r"\\multicolumn\{(\d+)\}", cell)
            width = int(match.group(1)) if match else 1
            if match:
                spans.append((column, column + width - 1))
            column += width
        if spans:
            rules = "".join(f"\\cmidrule(lr){{{a}-{b}}}" for a, b in spans)
            lines.insert(i + 1, rules)
        break
    return "\n".join(lines)


def wrap_table(tabular: str, caption: str, label: str, placement: str = "htbp") -> str:
    """Wrap a bare `tabular` environment in a centred float with the caption below.

    `\\label` must follow `\\caption`: LaTeX's `\\ref` resolves to whatever
    counter was last stepped, and it is `\\caption` that steps the table
    counter. A label placed before the caption silently points at the
    *previous* float instead.
    """
    body = tabular.strip()
    # pandas emits the float wrapper too when it is given a caption; strip any
    # stray one so this function is the only thing producing \begin{table}.
    body = re.sub(r"^\\begin\{table\}[^\n]*\n", "", body)
    body = re.sub(r"\n\\end\{table\}$", "", body).strip()
    body = _tidy_multiindex_header(body)
    if not body.startswith("\\begin{tabular}"):
        raise ValueError(
            f"expected a bare tabular environment, got: {body[:80]!r}"
        )
    return (
        f"\\begin{{table}}[{placement}]\n"
        f"\\centering\n"
        f"{body}\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"\\end{{table}}\n"
    )
