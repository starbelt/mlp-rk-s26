#!/usr/bin/env python3
# rewrite_dt_s.py
#
# Usage:
#   python3 rewrite_dt_s.py in.csv out.csv --dt_s_avg 0.1
#
# Copies in.csv to out.csv but replaces dt_s with dur_s/max_points,
# where max_points = int(max(dur_s)/dt_s_avg).

import argparse
import csv

def rewrite_dt_s(in_csv: str, out_csv: str, dt_s_avg: float) -> None:
    # Read all rows
    with open(in_csv, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError("Input CSV appears to have no header row.")
        if "dur_s" not in fieldnames or "dt_s" not in fieldnames:
            raise ValueError("Input CSV must contain 'dur_s' and 'dt_s' columns.")

        rows = list(reader)

    if not rows:
        raise ValueError("Input CSV has a header but no data rows.")

    # Compute max_points from max(dur_s)
    durs = []
    for i, r in enumerate(rows):
        try:
            durs.append(float(r["dur_s"]))
        except Exception as e:
            raise ValueError(f"Row {i+1}: bad dur_s value {r.get('dur_s')!r}") from e

    max_dur_s = max(durs)
    max_points = int(max_dur_s / dt_s_avg)

    if max_points <= 0:
        raise ValueError(
            f"Computed max_points={max_points}. Check dt_s_avg={dt_s_avg} and dur_s values."
        )

    # Replace dt_s for each row
    for i, r in enumerate(rows):
        dur = float(r["dur_s"])
        new_dt = dur / max_points
        # Keep as a reasonable float string; adjust formatting if you want
        r["dt_s"] = f"{new_dt:.10g}"

    # Write output CSV
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {out_csv}")
    print(f"max_dur_s={max_dur_s}, dt_s_avg={dt_s_avg}, max_points={max_points}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_csv")
    ap.add_argument("out_csv")
    ap.add_argument("--dt_s_avg", type=float, default=0.1,
                    help="Mean timestep used to set max_points (default: 0.1)")
    args = ap.parse_args()
    rewrite_dt_s(args.in_csv, args.out_csv, args.dt_s_avg)

if __name__ == "__main__":
    main()