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

ACC_TOL = 0.2  # points
ECE_TOL = 0.3  # points


def compare(replay_dir, reference_dir, acc_tol=ACC_TOL, ece_tol=ECE_TOL) -> list[dict]:
    replay_dir, reference_dir = Path(replay_dir), Path(reference_dir)
    rows = []
    for path in sorted(replay_dir.rglob("*.json")):
        rel = path.relative_to(replay_dir)
        replay = json.loads(path.read_text(encoding="utf-8"))
        row = {"record": str(rel), "dataset": replay["dataset"], "method": replay["method"],
               "seed": replay.get("seed"), "acc_replay": 100 * replay["accuracy"],
               "ece_replay": 100 * replay["ece"]}
        ref_path = reference_dir / rel
        if not ref_path.exists():
            row.update(status="no reference")
            rows.append(row)
            continue
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        row.update(acc_ref=100 * ref["accuracy"], ece_ref=100 * ref["ece"])
        row["d_acc"] = row["acc_replay"] - row["acc_ref"]
        row["d_ece"] = row["ece_replay"] - row["ece_ref"]
        same_n = ref.get("n_samples") == replay.get("n_samples")
        ok = same_n and abs(row["d_acc"]) <= acc_tol and abs(row["d_ece"]) <= ece_tol
        row["status"] = "ok" if ok else ("N MISMATCH" if not same_n else "OUT OF TOLERANCE")
        rows.append(row)
    return rows


def format_rows(rows) -> str:
    head = f"{'record':40s} {'acc ref':>8s} {'replay':>8s} {'d':>6s} {'ece ref':>8s} {'replay':>8s} {'d':>6s}  status"
    lines = [head, "-" * len(head)]
    for r in rows:
        if "acc_ref" in r:
            lines.append(f"{r['record']:40s} {r['acc_ref']:8.2f} {r['acc_replay']:8.2f} {r['d_acc']:+6.2f} "
                         f"{r['ece_ref']:8.2f} {r['ece_replay']:8.2f} {r['d_ece']:+6.2f}  {r['status']}")
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
