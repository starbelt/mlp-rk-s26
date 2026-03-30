#!/usr/bin/env python3

import os
import csv
import math
import time
import argparse

IRR_W_P_M2 = 1366.1


def calc_solar_current(irr_w_m2, sa_m2, eff, vmp):
    return (irr_w_m2 * sa_m2 * eff) / vmp if vmp > 0.0 else 0.0


def calc_node_discr(q_c, c_f, i_a, esr_ohm, power_w):
    return (q_c / c_f + i_a * esr_ohm) ** 2 - 4.0 * power_w * esr_ohm


def calc_node_voltage(disc, q_c, c_f, i_a, esr_ohm):
    return 0.5 * ((q_c / c_f) + i_a * esr_ohm + math.sqrt(disc))


def calc_charge_at_voltage(target_v, c_f, i_a, esr_ohm, power_w):
    if target_v <= 0.0:
        return 0.0
    return c_f * (target_v + (power_w * esr_ohm) / target_v - i_a * esr_ohm)


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def parse_row(r):
    def gf(*keys):
        for k in keys:
            if k in r and r[k] != "":
                return float(r[k])
        raise KeyError(f"Missing keys: {keys}")

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


def run_rk2_timing_only(params):
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
    qt_c = q0_c
    p_mode_w = 0.0
    i1_a = calc_solar_current(IRR_W_P_M2, sa_m2, eff, vmp)

    disc = calc_node_discr(qt_c, c_f, i1_a, esr_ohm, p_mode_w)
    if disc < 0.0:
        p_mode_w = 0.0
        disc = calc_node_discr(qt_c, c_f, i1_a, esr_ohm, p_mode_w)
        
    node_v = calc_node_voltage(disc, qt_c, c_f, i1_a, esr_ohm)

    if vmp <= node_v and i1_a > 0.0:
        i1_a = 0.0
        disc = calc_node_discr(qt_c, c_f, i1_a, esr_ohm, p_mode_w)
        if disc < 0.0:
            p_mode_w = 0.0
            disc = calc_node_discr(qt_c, c_f, i1_a, esr_ohm, p_mode_w)
        node_v = calc_node_voltage(disc, qt_c, c_f, i1_a, esr_ohm)

    if node_v <= vlo and p_mode_w != 0.0:
        p_mode_w = 0.0
        disc = calc_node_discr(qt_c, c_f, i1_a, esr_ohm, p_mode_w)
        node_v = calc_node_voltage(disc, qt_c, c_f, i1_a, esr_ohm)

    time_on_accum = 0.0
    t_start = time.perf_counter()
    n_steps = 0

    while t_s < dur_s:
        t_old_s = t_s
        t_s += dt_s
        n_steps += 1

        q_old_c = qt_c
        node_v_old = node_v
        p_old_w = p_mode_w

        i1_fixed = i1_a

        i3_k1 = 0.0
        if node_v > 0.0 and p_mode_w > 0.0:
            i3_k1 = p_mode_w / node_v

        k1 = i1_fixed - i3_k1

        qt_mid = qt_c + 0.5 * dt_s * k1
        if qt_mid < 0.0:
            qt_mid = 0.0

        disc_mid = calc_node_discr(qt_mid, c_f, i1_fixed, esr_ohm, p_mode_w)

        p_mid = p_mode_w
        if disc_mid < 0.0:
            p_mid = 0.0
            disc_mid = calc_node_discr(qt_mid, c_f, i1_fixed, esr_ohm, p_mid)

        node_v_mid = calc_node_voltage(disc_mid, qt_mid, c_f, i1_fixed, esr_ohm)

        i3_k2 = 0.0
        if node_v_mid > 0.0 and p_mid > 0.0:
            i3_k2 = p_mid / node_v_mid

        k2 = i1_fixed - i3_k2

        qt_c = qt_c + dt_s * k2
        if qt_c < 0.0:
            qt_c = 0.0
        q_tent_c = qt_c

        i1_a = calc_solar_current(IRR_W_P_M2, sa_m2, eff, vmp)
        i1_fixed = i1_a

        disc = calc_node_discr(qt_c, c_f, i1_fixed, esr_ohm, p_mode_w)
        if disc < 0.0:
            p_mode_w = 0.0
            disc = calc_node_discr(qt_c, c_f, i1_a, esr_ohm, p_mode_w)

        node_v = calc_node_voltage(disc, qt_c, c_f, i1_a, esr_ohm)

        if vmp <= node_v and i1_a > 0.0:
            i1_a = 0.0
            disc = calc_node_discr(qt_c, c_f, i1_a, esr_ohm, p_mode_w)
            if disc < 0.0:
                p_mode_w = 0.0
                disc = calc_node_discr(qt_c, c_f, i1_a, esr_ohm, p_mode_w)
            node_v = calc_node_voltage(disc, qt_c, c_f, i1_a, esr_ohm)

        vlo_on_fraction = 1.0
        crossed_vlo = (
            p_old_w > 0.0 and
            node_v_old > vlo and
            node_v <= vlo
        )
        if crossed_vlo:
            q_cross_c = max(calc_charge_at_voltage(vlo, c_f, i1_fixed, esr_ohm, p_old_w), 0.0)
            denom_c = q_tent_c - q_old_c
            if abs(denom_c) > 1e-18:
                lam = (q_cross_c - q_old_c) / denom_c
            else:
                lam = 1.0
            lam = max(0.0, min(1.0, lam))
            dt_rem_s = (1.0 - lam) * dt_s

            qt_c = max(q_cross_c + i1_fixed * dt_rem_s, 0.0)
            p_mode_w = 0.0

            disc = calc_node_discr(qt_c, c_f, i1_fixed, esr_ohm, p_mode_w)
            node_v = calc_node_voltage(disc, qt_c, c_f, i1_fixed, esr_ohm)
            vlo_on_fraction = lam

        if p_mode_w == 0.0 and node_v >= vhi:
            p_mode_w = p_on_w
        if node_v <= vlo and p_mode_w != 0.0:
            p_mode_w = 0.0

        if p_old_w > 0.0:
            time_on_accum += dt_s * vlo_on_fraction

    exec_time_total_s = time.perf_counter() - t_start
    exec_time_per_step_s = exec_time_total_s / max(n_steps, 1)
    duty_cycle_percent = 100.0 * time_on_accum / max(dur_s, 1e-18)
    duty_cycle_percent = float(max(0.0, min(100.0, duty_cycle_percent)))

    return exec_time_total_s, exec_time_per_step_s, n_steps, duty_cycle_percent


def append_summary_csv(path, row):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    file_exists = os.path.exists(path)

    with open(path, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "row_idx",
                #"num_runs",
                "dt_s",
                #"dur_s",
                #"n_steps",
                #"avg_total_runtime_s",
                "avg_per_step_runtime_s",
                "min_total_runtime_s",
                "max_total_runtime_s",
                #"avg_duty_cycle_percent",
            ])

        writer.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--row_idx", type=int, default=0)
    ap.add_argument("--num_runs", type=int, default=100)
    ap.add_argument("--output", type=str, required=True)
    args = ap.parse_args()

    rows = read_rows(args.input)
    if len(rows) == 0:
        raise ValueError(f"No rows found in input CSV: {args.input}")

    if args.row_idx < 0 or args.row_idx >= len(rows):
        raise ValueError(f"row_idx out of range: {args.row_idx} (rows={len(rows)})")

    params = parse_row(rows[args.row_idx])

    total_times = []
    step_times = []
    duty_cycles = []

    for _ in range(args.num_runs):
        total_s, per_step_s, n_steps, duty_pct = run_rk2_timing_only(params)
        total_times.append(total_s)
        step_times.append(per_step_s)
        duty_cycles.append(duty_pct)

    avg_total = sum(total_times) / len(total_times)
    avg_step = sum(step_times) / len(step_times)
    min_total = min(total_times)
    max_total = max(total_times)
    avg_duty = sum(duty_cycles) / len(duty_cycles)

    append_summary_csv(args.output, [
        args.row_idx,
       # args.num_runs,
        f"{params['dt_s']:.12e}",
       # f"{params['dur_s']:.12e}",
       # n_steps,
        f"{avg_total:.12e}",
       # f"{avg_step:.12e}",
        f"{min_total:.12e}",
        f"{max_total:.12e}",
        # f"{avg_duty:.12e}",
    ])

    #print(
        #f"[RK2 row {args.row_idx}] "
        #f"avg_total={avg_total:.6e} s, "
        #f"avg_step={avg_step:.6e} s, "
        #f"steps={n_steps}, "
        #f"avg_duty={avg_duty:.3f}%"
    #)


if __name__ == "__main__":
    main()