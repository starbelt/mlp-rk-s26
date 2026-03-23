#!/usr/bin/env python3

import os
import csv
import math
import time
import argparse

IRR_W_P_M2 = 1366.1


# ---------------- Core helpers ----------------
def calc_solar_current(irr_w_m2, sa_m2, eff, vmp):
    return (irr_w_m2 * sa_m2 * eff) / vmp if vmp > 0.0 else 0.0


def calc_node_discr(q_c, c_f, i_a, esr_ohm, power_w):
    return (q_c / c_f + i_a * esr_ohm) ** 2 - 4.0 * power_w * esr_ohm


def calc_node_voltage(disc, q_c, c_f, i_a, esr_ohm):
    return 0.5 * ((q_c / c_f) + i_a * esr_ohm + math.sqrt(disc))


# ---------------- CSV helpers ----------------
def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def parse_row(r):
    def gf(key):
        if key not in r:
            raise KeyError(f"Missing key: {key}")
        return float(r[key])

    return dict(
        sa_m2=gf("sa_m2"),
        eff=gf("eff"),
        vmp=gf("vmp"),
        c_f=gf("c_f"),
        esr_ohm=gf("esr_ohm"),
        q0_c=gf("q0_c"),
        p_on_w=gf("p_on_w"),
        vhi=gf("vhi"),
        vlo=gf("vlo"),
        dt_s=gf("dt_s"),
        dur_s=gf("dur_s"),
    )


# ---------------- Simulation ----------------
def run_rk1_timing_only(params):
    sa_m2 = params["sa_m2"]
    eff = params["eff"]
    vmp = params["vmp"]
    c_f = params["c_f"]
    esr_ohm = params["esr_ohm"]
    q0_c = params["q0_c"]
    p_on_w = params["p_on_w"]
    vhi = params["vhi"]
    vlo = params["vlo"]
    dt_s = params["dt_s"]
    dur_s = params["dur_s"]

    t_s = 0.0
    imp_a = calc_solar_current(IRR_W_P_M2, sa_m2, eff, vmp)
    i1_a = imp_a
    qt_c = q0_c
    p_mode_w = 0.0

    node_discr = calc_node_discr(qt_c, c_f, i1_a, esr_ohm, p_mode_w)
    if node_discr < 0.0:
        p_mode_w = 0.0
        node_discr = calc_node_discr(qt_c, c_f, i1_a, esr_ohm, p_mode_w)

    node_v = calc_node_voltage(node_discr, qt_c, c_f, i1_a, esr_ohm)

    if vmp <= node_v and i1_a > 0.0:
        i1_a = 0.0

    if node_v <= vlo and p_mode_w != 0.0:
        p_mode_w = 0.0

    n_steps = 0

    start = time.perf_counter()

    while t_s < dur_s:
        t_s += dt_s
        n_steps += 1

        i3_a = 0.0
        if node_v > 0.0 and p_mode_w > 0.0:
            i3_a = p_mode_w / node_v

        k1 = i1_a - i3_a
        qt_c += dt_s * k1
        if qt_c < 0.0:
            qt_c = 0.0

        i1_nom = calc_solar_current(IRR_W_P_M2, sa_m2, eff, vmp)
        disc = calc_node_discr(qt_c, c_f, i1_nom, esr_ohm, p_mode_w)

        if disc < 0.0:
            p_mode_w = 0.0
            disc = calc_node_discr(qt_c, c_f, i1_nom, esr_ohm, p_mode_w)

        node_v = calc_node_voltage(disc, qt_c, c_f, i1_nom, esr_ohm)

        if p_mode_w == 0.0 and node_v >= vhi:
            p_mode_w = p_on_w
            if calc_node_discr(qt_c, c_f, i1_nom, esr_ohm, p_mode_w) < 0.0:
                p_mode_w = 0.0

        if vmp <= node_v and i1_nom > 0.0:
            i1_a = 0.0
            if calc_node_discr(qt_c, c_f, i1_a, esr_ohm, p_mode_w) < 0.0:
                p_mode_w = 0.0
        else:
            i1_a = i1_nom

        if p_mode_w > 0.0 and node_v <= vlo:
            p_mode_w = 0.0

    total_time = time.perf_counter() - start
    avg_step_time = total_time / max(n_steps, 1)

    return total_time, avg_step_time, n_steps


# ---------------- CSV writing (FIXED) ----------------
def append_summary_csv(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    file_exists = os.path.exists(path)

    with open(path, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "row_idx",
                "dt_s",
                "avg_per_step_runtime_s",
                "min_total_runtime_s",
                "max_total_runtime_s",
            ])

        writer.writerow(row)


# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--row_idx", type=int, default=0)
    parser.add_argument("--num_runs", type=int, default=100)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    rows = read_rows(args.input)
    params = parse_row(rows[args.row_idx])

    total_times = []
    step_times = []

    for _ in range(args.num_runs):
        total_s, per_step_s, n_steps = run_rk1_timing_only(params)
        total_times.append(total_s)
        step_times.append(per_step_s)

    avg_total = sum(total_times) / len(total_times)
    avg_step = sum(step_times) / len(step_times)
    min_total = min(total_times)
    max_total = max(total_times)

    append_summary_csv(args.output, [
        args.row_idx,
        args.num_runs,
        f"{params['dt_s']:.12e}",
        f"{params['dur_s']:.12e}",
        n_steps,
        f"{avg_total:.12e}",
        f"{avg_step:.12e}",
        f"{min_total:.12e}",
        f"{max_total:.12e}",
    ])

    #print(f"[Row {args.row_idx}] avg={avg_total:.3e} s")


if __name__ == "__main__":
    main()