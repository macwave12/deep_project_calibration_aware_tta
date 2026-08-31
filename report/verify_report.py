"""Structural verification of report/main.tex, since pdflatex is not installed here.

Checks, mechanically (no eyeballing):
  1. Every `\\input{...}` target resolves to a file that exists (relative to report/,
     matching how pdflatex would resolve it when invoked as `cd report; pdflatex main.tex`).
  2. Every `\\includegraphics[...]{...}` target resolves to a real PNG, respecting the
     `\\graphicspath{{figures/}}` declared in main.tex.
  3. `\\bibliography{refs}` points at a real .bib file, and every `\\cite`/`\\citet`/`\\citep`
     key (comma-separated keys supported) appears as an `@entrytype{key,` in it.
  4. Braces balance (ignoring `\\{` and `\\}` literal-brace escapes) in every file reached
     by `\\input` from main.tex.
  5. `\\begin{env}`/`\\end{env}` pairs balance and nest correctly within every file reached
     by `\\input` from main.tex (checked per-file, since each `\\input` file's `\\begin`s
     are expected to close within that same file).
  6. Every `\\command` token used anywhere in the document resolves to something the
     LaTeX kernel/article class provides, something one of the `\\usepackage`s actually
     declared in the preamble provides, or something the document defines itself via
     `\\newcommand`/`\\DeclareMathOperator`. This is the check that would have caught
     `\\argmin` (needs `\\DeclareMathOperator*`), `\\mathbb` (needs `amssymb`, and even
     then `\\mathbb{1}` is wrong -- blackboard-bold digits don't exist in that font) and
     `\\citet` (needs `natbib`) before they reached a real `pdflatex` run.
  7. Every `\\ref{...}`/`\\eqref{...}` target has a matching `\\label{...}` somewhere in the
     document. An undefined reference does not stop `pdflatex` -- it renders as a literal
     `??` in the PDF with a build warning -- but that failure mode is exactly the kind a
     script substituting for a real compile should still catch rather than silently miss.

This is a whitelist-based approximation of a real LaTeX engine, not a real one: it cannot
prove the document compiles, only that these specific, previously-real classes of failure
are absent. It has a curated (not exhaustive) notion of what base LaTeX/article/amsmath
provide "for free" -- see BASE_COMMANDS below -- so an exotic command from a package we do
not use here could still slip past undetected. `--dir` overrides which report directory is
checked (used to prove this check fails on a deliberately pre-fix copy; defaults to this
script's own directory, i.e. the real, current report/).

Exits non-zero if anything fails, and prints a PASS/FAIL summary either way.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

INPUT_RE = re.compile(r"\\input\{([^}]+)\}")
INCLUDEGRAPHICS_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
BIBLIOGRAPHY_RE = re.compile(r"\\bibliography\{([^}]+)\}")
CITE_RE = re.compile(r"\\cite[tp]?\*?(?:\[[^\]]*\])?(?:\[[^\]]*\])?\{([^}]+)\}")
BIB_ENTRY_RE = re.compile(r"@\w+\{\s*([^,\s]+)\s*,")
BEGIN_RE = re.compile(r"\\begin\{([^}]+)\}")
END_RE = re.compile(r"\\end\{([^}]+)\}")
USEPACKAGE_RE = re.compile(r"\\usepackage(?:\[[^\]]*\])?\{([^}]+)\}")
SELF_DEFINE_RE = re.compile(
    r"\\(?:newcommand|renewcommand|DeclareMathOperator)\*?\{?\\([a-zA-Z]+)\}?"
)
COMMAND_USE_RE = re.compile(r"\\([a-zA-Z]+)")
DOCUMENT_START_RE = re.compile(r"\\begin\{document\}")
LABEL_RE = re.compile(r"\\label\{([^}]+)\}")
REF_RE = re.compile(r"\\(?:eq)?ref\{([^}]+)\}")

# What the LaTeX kernel + the `article` class + amsmath (loaded unconditionally by this
# report) provide without any further package. Not exhaustive -- curated to what a report
# like this one plausibly uses -- so this is a whitelist, not a LaTeX implementation.
BASE_COMMANDS = {
    "documentclass", "usepackage", "begin", "end", "title", "author", "date", "maketitle",
    "section", "subsection", "subsubsection", "paragraph", "item", "label", "ref", "pageref",
    "input", "include", "emph", "textbf", "textit", "textrm", "texttt", "footnote", "caption",
    "multicolumn", "hline", "cline",
    "today", "appendix", "centering", "bibliographystyle", "bibliography", "cite", "and",
    "newpage", "noindent", "par", "LaTeX", "TeX", "hspace", "vspace", "linewidth", "textwidth",
    "newcommand", "renewcommand", "DeclareMathOperator",
    # Spacing and line-breaking primitives from the TeX kernel:
    "hfill", "vfill", "allowbreak", "linebreak", "newline", "textbackslash",
    # amsmath, always loaded by this report's preamble:
    "frac", "sum", "prod", "int", "left", "right", "text", "substack", "operatorname",
    "boldsymbol", "nonumber", "notag", "eqref",
    # math-mode symbols available in plain LaTeX math without any extra package:
    "alpha", "beta", "gamma", "delta", "epsilon", "sigma", "times", "cdot", "ldots", "dots",
    "infty", "rightarrow", "leftarrow", "leq", "geq", "in", "notin", "bar", "hat", "tilde",
    "vec", "star", "mathrm", "mathbf", "mathit", "mathcal", "approx", "exp", "log", "sim",
    "propto", "neq", "pm",
}

# Commands each optional package adds, gated on that package actually being \usepackage'd.
# Only lists what this report plausibly uses -- a real LaTeX distribution's real command
# set per package is much larger; this whitelist exists to catch *our* three known failure
# classes and similar ones, not to be a general-purpose LaTeX linter.
PACKAGE_COMMANDS = {
    "graphicx": {"includegraphics", "graphicspath", "rotatebox", "scalebox", "resizebox"},
    "booktabs": {"toprule", "midrule", "bottomrule", "addlinespace", "cmidrule", "specialrule"},
    "amssymb": {"mathbb", "gtrsim", "lesssim", "leqslant", "geqslant", "nleq", "ngeq"},
    "hyperref": {"url", "href", "autoref", "nameref"},
    "caption": {"captionof", "captionsetup"},
    "subcaption": {"subcaption", "subfloat"},
    "natbib": {"citet", "citep", "citealt", "citealp", "citeauthor", "citeyear",
               "bibpunct", "setcitestyle"},
    "geometry": set(),
    "amsmath": set(),  # already folded into BASE_COMMANDS above; listed so it's a known package
}


def strip_comments(text: str) -> str:
    """Drop everything from an unescaped % to end-of-line, so a % description in a
    guidance comment block (this report's sections all start with one) cannot be
    mistaken for real \\input/\\includegraphics/\\cite/macro usage."""
    out_lines = []
    for line in text.split("\n"):
        out = []
        i = 0
        while i < len(line):
            if line[i] == "%" and (i == 0 or line[i - 1] != "\\"):
                break
            out.append(line[i])
            i += 1
        out_lines.append("".join(out))
    return "\n".join(out_lines)


class Verifier:
    def __init__(self, report_dir: Path):
        self.report_dir = report_dir
        self.main_tex = report_dir / "main.tex"
        self.errors: list[str] = []
        self.info: list[str] = []

    def rel_label(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.report_dir))
        except ValueError:
            return str(path.relative_to(self.report_dir.parent))

    def resolve(self, relative: str, base: Path | None = None) -> Path:
        base = base if base is not None else self.report_dir
        path = (base / relative).resolve()
        if path.suffix == "":
            path = path.with_suffix(".tex")
        return path

    def check_braces(self, text: str, label: str) -> None:
        stripped = re.sub(r"\\[{}]", "", text)
        depth = 0
        for i, ch in enumerate(stripped):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth < 0:
                    self.errors.append(f"{label}: unmatched closing brace at offset {i}")
                    return
        if depth != 0:
            self.errors.append(f"{label}: brace imbalance, net depth {depth} (expected 0)")
        else:
            self.info.append(f"{label}: braces balance ({stripped.count('{')} pairs)")

    def check_environments(self, text: str, label: str) -> None:
        stack: list[str] = []
        tokens = [(m.start(), "begin", m.group(1)) for m in BEGIN_RE.finditer(text)]
        tokens += [(m.start(), "end", m.group(1)) for m in END_RE.finditer(text)]
        tokens.sort(key=lambda t: t[0])
        for _, kind, env in tokens:
            if kind == "begin":
                stack.append(env)
            else:
                if not stack:
                    self.errors.append(f"{label}: \\end{{{env}}} with no open environment")
                    continue
                top = stack.pop()
                if top != env:
                    self.errors.append(
                        f"{label}: \\begin{{{top}}} closed by mismatched \\end{{{env}}}"
                    )
        if stack:
            self.errors.append(f"{label}: unclosed environment(s): {stack}")
        else:
            self.info.append(f"{label}: {len(tokens)//2} begin/end environment(s) balance")

    def check_refs(self, all_text: str) -> None:
        labels = set(LABEL_RE.findall(all_text))
        refs = set(REF_RE.findall(all_text))
        missing = sorted(refs - labels)
        if missing:
            self.errors.append(f"\\ref/\\eqref target(s) with no matching \\label: {missing}")
        self.info.append(
            f"\\ref/\\eqref targets: {len(refs)} used, {len(labels)} \\label(s) defined, "
            f"{len(missing)} unresolved"
        )

    def check_undefined_macros(self, main_text: str, all_input_text: list[tuple[Path, str]]) -> None:
        preamble_end = DOCUMENT_START_RE.search(main_text)
        preamble = main_text[: preamble_end.start()] if preamble_end else main_text

        loaded_packages: set[str] = set()
        for group in USEPACKAGE_RE.findall(preamble):
            loaded_packages.update(p.strip() for p in group.split(","))
        self.info.append(f"packages loaded: {sorted(loaded_packages)}")

        allowed = set(BASE_COMMANDS)
        unknown_packages = []
        for pkg in loaded_packages:
            if pkg in PACKAGE_COMMANDS:
                allowed.update(PACKAGE_COMMANDS[pkg])
            else:
                unknown_packages.append(pkg)
        if unknown_packages:
            self.info.append(
                f"packages loaded but not in this checker's whitelist (assumed OK, "
                f"not verified): {sorted(unknown_packages)}"
            )

        all_text = main_text + "\n" + "\n".join(t for _, t in all_input_text)
        self_defined = set(SELF_DEFINE_RE.findall(all_text))
        allowed.update(self_defined)
        if self_defined:
            self.info.append(f"self-defined commands found: {sorted(self_defined)}")

        used = set(COMMAND_USE_RE.findall(all_text))
        unknown = sorted(used - allowed)
        if unknown:
            self.errors.append(
                f"possibly-undefined command(s) (not base LaTeX, not provided by a loaded "
                f"package, not self-defined): {unknown}"
            )
        self.info.append(
            f"macro check: {len(used)} distinct \\command tokens used, "
            f"{len(unknown)} not accounted for by loaded packages/self-definitions"
        )

    def run(self) -> int:
        if not self.main_tex.exists():
            print(f"FATAL: {self.main_tex} does not exist")
            return 2

        main_text = strip_comments(self.main_tex.read_text(encoding="utf-8"))

        # --- 1. \input targets ---------------------------------------------
        # Recursive: main.tex \input's the eight section files, and results.tex
        # itself \input's two generated tables (results/summary_table.tex,
        # tables/ablation.tex). Every level must resolve.
        all_input_targets: list[str] = []
        n_input_ok = 0
        input_files: list[Path] = []
        to_scan = [("main.tex", main_text)]
        seen_targets: set[str] = set()
        while to_scan:
            source_label, source_text = to_scan.pop(0)
            for target in INPUT_RE.findall(source_text):
                if target in seen_targets:
                    continue
                seen_targets.add(target)
                all_input_targets.append(target)
                path = self.resolve(target)
                if path.exists():
                    n_input_ok += 1
                    input_files.append(path)
                    child_text = strip_comments(path.read_text(encoding="utf-8"))
                    to_scan.append((target, child_text))
                else:
                    self.errors.append(
                        f"\\input{{{target}}} (from {source_label}) -> {path} does NOT exist"
                    )
        self.info.append(
            f"\\input targets (recursive): {n_input_ok}/{len(all_input_targets)} resolve "
            f"({all_input_targets})"
        )

        # --- brace / environment balance ------------------------------------
        self.check_braces(main_text, "main.tex")
        self.check_environments(main_text, "main.tex")
        all_input_text: list[tuple[Path, str]] = []
        for path in input_files:
            text = strip_comments(path.read_text(encoding="utf-8"))
            all_input_text.append((path, text))
            self.check_braces(text, self.rel_label(path))
            self.check_environments(text, self.rel_label(path))

        # --- 2. \includegraphics targets ------------------------------------
        graphicspath_candidates = [self.report_dir, self.report_dir / "figures"]
        all_tex_text = main_text + "\n" + "\n".join(t for _, t in all_input_text)
        graphics_targets = INCLUDEGRAPHICS_RE.findall(all_tex_text)
        n_graphics_ok = 0
        for target in graphics_targets:
            found = None
            for base in graphicspath_candidates:
                candidate = (base / target).resolve()
                if candidate.exists():
                    found = candidate
                elif candidate.suffix == "":
                    for ext in (".png", ".pdf", ".jpg", ".jpeg"):
                        alt = candidate.with_suffix(ext)
                        if alt.exists():
                            found = alt
                            break
                if found:
                    break
            if found is not None:
                n_graphics_ok += 1
            else:
                self.errors.append(
                    f"\\includegraphics{{{target}}} does not resolve under {graphicspath_candidates}"
                )
        self.info.append(
            f"\\includegraphics targets: {n_graphics_ok}/{len(graphics_targets)} resolve "
            f"({graphics_targets})"
        )

        # --- 3. bibliography + \cite keys -----------------------------------
        bib_matches = BIBLIOGRAPHY_RE.findall(all_tex_text)
        if not bib_matches:
            self.errors.append("no \\bibliography{...} found in main.tex")
        else:
            bib_name = bib_matches[0]
            bib_path = self.report_dir / f"{bib_name}.bib"
            if not bib_path.exists():
                self.errors.append(f"\\bibliography{{{bib_name}}} -> {bib_path} does NOT exist")
            else:
                bib_text = bib_path.read_text(encoding="utf-8")
                bib_keys = set(BIB_ENTRY_RE.findall(bib_text))
                self.info.append(f"refs.bib: {len(bib_keys)} entries ({sorted(bib_keys)})")

                cite_keys: set[str] = set()
                for m in CITE_RE.finditer(all_tex_text):
                    for key in m.group(1).split(","):
                        cite_keys.add(key.strip())
                missing = sorted(k for k in cite_keys if k not in bib_keys)
                if missing:
                    self.errors.append(f"\\cite key(s) not found in {bib_name}.bib: {missing}")
                self.info.append(
                    f"\\cite keys used: {len(cite_keys)} ({sorted(cite_keys)}); "
                    f"all resolve: {not missing}"
                )

        # --- 6. undefined-macro check ----------------------------------------
        self.check_undefined_macros(main_text, all_input_text)

        # --- 7. \ref/\eqref resolution ----------------------------------------
        self.check_refs(all_tex_text)

        # --- summary -----------------------------------------------------------
        print("=" * 70)
        print(f"STRUCTURAL VERIFICATION OF {self.rel_label(self.main_tex) if self.main_tex.is_relative_to(self.report_dir) else self.main_tex}")
        print("=" * 70)
        for line in self.info:
            print(f"  OK   {line}")
        print("-" * 70)
        if self.errors:
            for line in self.errors:
                print(f"  FAIL {line}")
            print("-" * 70)
            print(f"RESULT: FAIL ({len(self.errors)} error(s))")
            return 1
        print(
            f"RESULT: PASS (0 errors; {len(all_input_targets)} \\input, {len(graphics_targets)} "
            f"\\includegraphics, bibliography+citations all resolved, braces/environments "
            f"balance, all macros accounted for)"
        )
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir", type=Path, default=Path(__file__).resolve().parent,
        help="report/ directory to check (default: this script's own directory)",
    )
    args = parser.parse_args()
    return Verifier(args.dir.resolve()).run()


if __name__ == "__main__":
    raise SystemExit(main())
