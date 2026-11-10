"""RULES §7 check: do CPU replay records reproduce the GPU run records?

Pairs every `<replay_dir>/**/<dataset>_<method>.json` with the record at the same
relative path under `<reference_dir>` (default `results/raw`, i.e. the records
behind results/summary.csv, ablation.csv and ablation_gate.csv) and reports
accuracy / ECE differences in points. Exit status 1 if any pair is outside the
tolerance (0.2 accuracy points, 0.3 ECE points) or nothing could be compared.
Records with no GPU counterpart (seeds 1/2, val, protocol P2) are listed as
"no reference" and do not fail the check.

    python analysis/compare_replay.py [--replay replay_results/raw] [--reference results/raw]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:  # runnable as `python analysis/compare_replay.py`
    sys.path.insert(0, str(_REPO))

ACC_TOL = 0.2  # points
ECE_TOL = 0.3  # points


# Not configuration: the replay's provenance block, and diagnostics the run
# produced (their values legitimately move with numerics). Everything else in
# `hyperparams` describes how the run was configured and must be identical.
IGNORED_HYPERPARAMS = {"replay", "temperature", "admission"}
# Keys added after the GPU records were written, with the value those runs had.
IMPLIED_DEFAULTS = {"protocol": "published"}


def _config_defaults() -> dict:
    from dataclasses import fields

    from online_method.cal_tda import CalTdaConfig

    return {f.name: f.default for f in fields(CalTdaConfig)}


def hyperparam_diff(reference: dict, replay: dict) -> dict:
    """{key: (reference value, replay value)} for every configuration key that differs.

    A GPU record written before a config field existed (e.g. `fusion_gate` in the
    first ablation) lacks that key; the run had the field's default behaviour,
    so a replay value equal to the `CalTdaConfig` default is not a difference.
    For the same reason `inert` is compared over the fields the record knew.
    """
    reference = {**IMPLIED_DEFAULTS, **(reference or {})}
    replay = {**IMPLIED_DEFAULTS, **(replay or {})}
    defaults = _config_defaults()
    known = set(reference) | set(reference.get("inert") or [])
    diff = {}
    for key in sorted((set(reference) | set(replay)) - IGNORED_HYPERPARAMS):
        ref_value, replay_value = reference.get(key), replay.get(key)
        if key == "inert":
            replay_value = [x for x in (replay_value or []) if x in known]
            ref_value = list(ref_value or [])
            if sorted(ref_value) != sorted(replay_value):
                diff[key] = (ref_value, replay_value)
            continue
        if key not in reference and key in defaults and replay_value == defaults[key]:
            continue  # field added after this record; the run had its default
        if ref_value != replay_value:
            diff[key] = (ref_value, replay_value)
    return diff


def reference_path(reference_dir: Path, rel: Path) -> Path:
    """`autocast_off/<x>` is the plain-fp32 A/B of `<x>`: same GPU reference."""
    parts = rel.parts
    if parts and parts[0] == "autocast_off":
        rel = Path(*parts[1:])
    return reference_dir / rel


def compare(replay_dir, reference_dir, acc_tol=ACC_TOL, ece_tol=ECE_TOL) -> list[dict]:
    replay_dir, reference_dir = Path(replay_dir), Path(reference_dir)
    rows = []
    for path in sorted(replay_dir.rglob("*.json")):
        rel = path.relative_to(replay_dir)
        replay = json.loads(path.read_text(encoding="utf-8"))
        row = {"record": rel.as_posix(), "dataset": replay["dataset"], "method": replay["method"],
               "seed": replay.get("seed"), "acc_replay": 100 * replay["accuracy"],
               "ece_replay": 100 * replay["ece"]}
        ref_path = reference_path(reference_dir, rel)
        if not ref_path.exists():
            row.update(status="no reference")
            rows.append(row)
            continue
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        row.update(acc_ref=100 * ref["accuracy"], ece_ref=100 * ref["ece"])
        row["d_acc"] = row["acc_replay"] - row["acc_ref"]
        row["d_ece"] = row["ece_replay"] - row["ece_ref"]
        row["hyperparam_diff"] = hyperparam_diff(ref.get("hyperparams"), replay.get("hyperparams"))
        same_n = ref.get("n_samples") == replay.get("n_samples")
        within = abs(row["d_acc"]) <= acc_tol and abs(row["d_ece"]) <= ece_tol
        if not same_n:
            row["status"] = "N MISMATCH"
        elif row["hyperparam_diff"]:
            row["status"] = "HYPERPARAMS DIFFER"
        else:
            row["status"] = "ok" if within else "OUT OF TOLERANCE"
        rows.append(row)
    return rows


def format_rows(rows) -> str:
    head = f"{'record':40s} {'acc ref':>8s} {'replay':>8s} {'d':>6s} {'ece ref':>8s} {'replay':>8s} {'d':>6s}  status"
    lines = [head, "-" * len(head)]
    for r in rows:
        if "acc_ref" in r:
            lines.append(f"{r['record']:40s} {r['acc_ref']:8.2f} {r['acc_replay']:8.2f} {r['d_acc']:+6.2f} "
                         f"{r['ece_ref']:8.2f} {r['ece_replay']:8.2f} {r['d_ece']:+6.2f}  {r['status']}")
            for key, (ref_value, replay_value) in r.get("hyperparam_diff", {}).items():
                lines.append(f"    hyperparams.{key}: reference {ref_value!r} vs replay {replay_value!r}")
        else:
            lines.append(f"{r['record']:40s} {'':8s} {r['acc_replay']:8.2f} {'':6s} {'':8s} "
                         f"{r['ece_replay']:8.2f} {'':6s}  {r['status']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--replay", default="replay_results/raw")
    p.add_argument("--reference", default="results/raw")
    a = p.parse_args(argv)
    rows = compare(a.replay, a.reference)
    if not rows:
        print(f"no replay records under {a.replay}")
        return 1
    print(format_rows(rows))
    compared = [r for r in rows if r["status"] != "no reference"]
    bad = [r for r in compared if r["status"] != "ok"]
    print(f"\n{len(compared) - len(bad)}/{len(compared)} compared records within tolerance "
          f"(accuracy {ACC_TOL} pts, ECE {ECE_TOL} pts); {len(rows) - len(compared)} without reference")
    return 1 if bad or not compared else 0


if __name__ == "__main__":
    sys.exit(main())
