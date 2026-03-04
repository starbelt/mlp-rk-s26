# sim_esys_cap_rk4.py
#
# Usage:
#   python3 scripts/sim_esys_cap_rk4.py --row_idx 0
#
# Writes:
#   outputs/rk4/r01/
#     log-node-v.csv      (FULL)
#     log-buff-v.csv      (FULL)
#     log-states.csv
#     meta.csv
#     opcount.csv
#     voltages_rk4.pdf    (downsampled visualization only)
# ------------------------------------------------------------

import os, csv, math, time, argparse
import matplotlib.pyplot as plt

IRR_W_P_M2 = 1366.1
PLOT_DOWNSAMPLE = 5  # ONLY for PDF visualization; CSV stays full.

# ---------------- Core helpers ----------------
def calc_solar_current(irr_w_m2, sa_m2, eff, vmp):
    # I_sun = (Irr * Area * eff) / Vmp
    return (irr_w_m2 * sa_m2 * eff) / vmp if vmp > 0.0 else 0.0

def calc_node_discr(q_c, c_f, i_a, esr_ohm, power_w):
    # (q/C + I*R)^2 - 4*P*R
    return (q_c / c_f + i_a * esr_ohm) ** 2 - 4.0 * power_w * esr_ohm

def calc_node_voltage(disc, q_c, c_f, i_a, esr_ohm):
    # v = 0.5 * (q/C + I*R + sqrt(disc))
    return 0.5 * ((q_c / c_f) + i_a * esr_ohm + math.sqrt(disc))


# ---------------- CSV helpers ----------------
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


# ---------------- Operation count ----------------
def write_opcount_csv(path, dt_s, dur_s, steps, opcounts):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "method","dt_s","dur_s","steps",
            "add","mul","div","sqrt","comp",
        ])
        w.writerow([
            "rk4", f"{dt_s:.12g}", f"{dur_s:.12g}", int(steps),
            int(opcounts["add"]), int(opcounts["mul"]), int(opcounts["div"]),
            int(opcounts["sqrt"]), int(opcounts["comp"]),
        ])


# ---------------- RK4 charge rate computation ----------------
def compute_charge_rate_at_point(q_c, i1_a, p_mode_w, c_f, esr_ohm, op):
    """
    dq/dt = i1 - i3, where i3 = P/V and V depends on q.
    """
    # disc = (q/C + I*R)^2 - 4*P*R
    op["div"] += 1; op["mul"] += 1; op["add"] += 1
    op["mul"] += 1; op["mul"] += 2; op["add"] += 1
    disc = calc_node_discr(q_c, c_f, i1_a, esr_ohm, p_mode_w)

    # If disc < 0, force load off for this derivative evaluation
    op["comp"] += 1
    p_used = p_mode_w
    if disc < 0.0:
        p_used = 0.0
        op["div"] += 1; op["mul"] += 1; op["add"] += 1
        op["mul"] += 1; op["mul"] += 2; op["add"] += 1
        disc = calc_node_discr(q_c, c_f, i1_a, esr_ohm, p_used)

    # node_v
    op["div"] += 1; op["mul"] += 1; op["add"] += 2; op["sqrt"] += 1; op["mul"] += 1
    node_v = calc_node_voltage(disc, q_c, c_f, i1_a, esr_ohm)

    # i3 = P/V
    i3 = 0.0
    op["comp"] += 2
    if node_v > 0.0 and p_used > 0.0:
        op["div"] += 1
        i3 = p_used / node_v

    op["add"] += 1
    return i1_a - i3


# ---------------- Final node voltage computation (end-of-step) ----------------
def compute_final_node_voltage(q_c, i1_a, p_mode_w, c_f, esr_ohm, op):
    # disc
    op["div"] += 1; op["mul"] += 1; op["add"] += 1
    op["mul"] += 1; op["mul"] += 2; op["add"] += 1
    disc = calc_node_discr(q_c, c_f, i1_a, esr_ohm, p_mode_w)

    # If disc < 0, force load off
    p_used = p_mode_w
    op["comp"] += 1
    if disc < 0.0:
        p_used = 0.0
        op["div"] += 1; op["mul"] += 1; op["add"] += 1
        op["mul"] += 1; op["mul"] += 2; op["add"] += 1
        disc = calc_node_discr(q_c, c_f, i1_a, esr_ohm, p_used)

    # node_v
    op["div"] += 1; op["mul"] += 1; op["add"] += 2; op["sqrt"] += 1; op["mul"] += 1
    node_v = calc_node_voltage(disc, q_c, c_f, i1_a, esr_ohm)

    return node_v, p_used, disc


# ---------------- Simulation (RK4 / Classic) ----------------
def run_rk4(params, timing_only: bool = False):
    sa_m2=params["sa_m2"]; eff=params["eff"]; vmp=params["vmp"]
    c_f=params["c_f"]; esr_ohm=params["esr_ohm"]; q0_c=params["q0_c"]
    p_on_w=params["p_on_w"]; vhi=params["vhi"]; vlo=params["vlo"]
    dt_s=params["dt_s"]; dur_s=params["dur_s"]

    op = dict(add=0, mul=0, div=0, sqrt=0, comp=0)

    t_s = 0.0

    # nominal solar current
    op["comp"] += 1
    if vmp > 0.0:
        op["mul"] += 2
        op["div"] += 1
    i1_a = calc_solar_current(IRR_W_P_M2, sa_m2, eff, vmp)

    qt_c = q0_c
    p_mode_w = 0.0

    # initial node voltage
    node_v, actual_power_w, _ = compute_final_node_voltage(qt_c, i1_a, p_mode_w, c_f, esr_ohm, op)
    p_mode_w = actual_power_w

    # Vmp clamp at init
    op["comp"] += 2
    if vmp <= node_v and i1_a > 0.0:
        i1_a = 0.0
        node_v, actual_power_w, _ = compute_final_node_voltage(qt_c, i1_a, p_mode_w, c_f, esr_ohm, op)
        if actual_power_w != p_mode_w:
            p_mode_w = actual_power_w

    # Vlo clamp at init
    op["comp"] += 1
    if node_v <= vlo:
        op["comp"] += 1
        if p_mode_w != 0.0:
            p_mode_w = 0.0
            node_v, _, _ = compute_final_node_voltage(qt_c, i1_a, p_mode_w, c_f, esr_ohm, op)

    # ---- logs ----
    # keep states always (for cycle timing)
    log_states = [[t_s, "OFF"]]
    # voltage logs
    log_node_v = [[t_s, node_v]]
    log_buff_v = [[t_s, qt_c / c_f]]

    # timing + duty
    time_on_accum = 0.0
    t_start = time.perf_counter()
    n_steps = 0

    while t_s < dur_s:
        op["add"] += 1
        t_s += dt_s
        n_steps += 1

        # Hold solar current fixed during RK4 stages
        i1_fixed = i1_a

        # k1
        k1 = compute_charge_rate_at_point(qt_c, i1_fixed, p_mode_w, c_f, esr_ohm, op)

        # q1 = q + 0.5*dt*k1
        op["mul"] += 1; op["mul"] += 1; op["add"] += 1
        q1 = qt_c + (dt_s * k1) * 0.5

        # k2
        k2 = compute_charge_rate_at_point(q1, i1_fixed, p_mode_w, c_f, esr_ohm, op)

        # q2 = q + 0.5*dt*k2
        op["mul"] += 1; op["mul"] += 1; op["add"] += 1
        q2 = qt_c + (dt_s * k2) * 0.5

        # k3
        k3 = compute_charge_rate_at_point(q2, i1_fixed, p_mode_w, c_f, esr_ohm, op)

        # q3 = q + dt*k3
        op["mul"] += 1; op["add"] += 1
        q3 = qt_c + dt_s * k3

        # k4
        k4 = compute_charge_rate_at_point(q3, i1_fixed, p_mode_w, c_f, esr_ohm, op)

        # s = k1 + 2*k2 + 2*k3 + k4
        op["mul"] += 2
        op["add"] += 3
        s = k1 + 2.0*k2 + 2.0*k3 + k4

        # q = q + dt*s/6
        op["mul"] += 1; op["div"] += 1; op["add"] += 1
        qt_c = qt_c + (dt_s * s) / 6.0

        # clamp q >= 0
        op["comp"] += 1
        if qt_c < 0.0:
            qt_c = 0.0

        # ---- end-of-step: update i1 and node voltage ----
        op["comp"] += 1
        if vmp > 0.0:
            op["mul"] += 2; op["div"] += 1
        i1_nom = calc_solar_current(IRR_W_P_M2, sa_m2, eff, vmp)

        node_v, p_used, _ = compute_final_node_voltage(qt_c, i1_nom, p_mode_w, c_f, esr_ohm, op)
        if p_used != p_mode_w:
            p_mode_w = p_used
            log_states.append([t_s, "power too high"])

        # VHI logic
        op["comp"] += 1
        if p_mode_w == 0.0:
            op["comp"] += 1
            if node_v >= vhi:
                p_mode_w = p_on_w
                log_states.append([t_s, "VHI"])
                op["div"] += 1; op["mul"] += 1; op["add"] += 1
                op["mul"] += 1; op["mul"] += 2; op["add"] += 1
                disc_check = calc_node_discr(qt_c, c_f, i1_nom, esr_ohm, p_mode_w)
                op["comp"] += 1
                if disc_check < 0.0:
                    p_mode_w = 0.0
                    log_states.append([t_s, "power too high"])

        # Vmp clamp
        op["comp"] += 2
        if vmp <= node_v and i1_nom > 0.0:
            i1_a = 0.0
            log_states.append([t_s, "node voltage too high"])
            op["div"] += 1; op["mul"] += 1; op["add"] += 1
            op["mul"] += 1; op["mul"] += 2; op["add"] += 1
            disc_check = calc_node_discr(qt_c, c_f, i1_a, esr_ohm, p_mode_w)
            op["comp"] += 1
            if disc_check < 0.0:
                p_mode_w = 0.0
                log_states.append([t_s, "power too high"])
        else:
            i1_a = i1_nom

        # VLO logic
        op["comp"] += 1
        if p_mode_w > 0.0:
            op["comp"] += 1
            if node_v <= vlo:
                p_mode_w = 0.0
                log_states.append([t_s, "VLO"])

        # duty
        op["comp"] += 1
        if p_mode_w > 0.0:
            op["add"] += 1
            time_on_accum += dt_s

        # voltage logs
        if not timing_only:
            log_node_v.append([t_s, node_v])
            log_buff_v.append([t_s, qt_c / c_f])

    exec_time_total_s = time.perf_counter() - t_start
    exec_time_per_step_s = exec_time_total_s / max(n_steps, 1)

    duty_cycle_percent = 100.0 * time_on_accum / max(dur_s, 1e-18)
    duty_cycle_percent = float(max(0.0, min(100.0, duty_cycle_percent)))

    # keep minimal end sample for timing_only
    if timing_only:
        if log_node_v[-1][0] != t_s:
            log_node_v.append([t_s, node_v])
        if log_buff_v[-1][0] != t_s:
            log_buff_v.append([t_s, qt_c / c_f])

    return (
        log_node_v,
        log_buff_v,
        log_states,
        exec_time_total_s,
        exec_time_per_step_s,
        n_steps,
        duty_cycle_percent,
        op,
    )


# ---------------- Write outputs ----------------
def write_outputs(out_dir, log_node_v, log_buff_v, log_states, meta_dict, params):
    os.makedirs(out_dir, exist_ok=True)

    node_csv = os.path.join(out_dir, "log-node-v.csv")
    buff_csv = os.path.join(out_dir, "log-buff-v.csv")

    with open(node_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "Node Voltage (V)"])
        for t, v in log_node_v:
            w.writerow([f"{t:.9f}", f"{v:.9f}"])

    with open(buff_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "Buffer Voltage (V)"])
        for t, v in log_buff_v:
            w.writerow([f"{t:.9f}", f"{v:.9f}"])

    with open(os.path.join(out_dir, "log-states.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "state"])
        for t, s in log_states:
            w.writerow([f"{t:.9f}", s])

    with open(os.path.join(out_dir, "meta.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key","value"])
        for k, v in meta_dict.items():
            w.writerow([k, v])

    dn = max(int(PLOT_DOWNSAMPLE), 1)
    node_ds = log_node_v[::dn]
    buff_ds = log_buff_v[::dn]
    t_plot  = [e[0] for e in node_ds]
    node_v  = [e[1] for e in node_ds]
    buff_v  = [e[1] for e in buff_ds]

    param_text = (
        f"sa_m2={params['sa_m2']:.4g}, eff={params['eff']:.3g}, vmp={params['vmp']:.3g}, "
        f"c_f={params['c_f']:.3g}, esr_ohm={params['esr_ohm']:.3g}, q0_c={params['q0_c']:.3g}, "
        f"p_on_w={params['p_on_w']:.3g}, vhi={params['vhi']:.3g}, vlo={params['vlo']:.3g}, "
        f"dt_s={params['dt_s']:.3g}, dur_s={params['dur_s']:.3g}"
    )

    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    fig.suptitle(param_text, fontsize=9, y=0.98)
    ax.plot(t_plot, node_v, marker=".", linestyle="None", markersize=2, label="Node Voltage (V)")
    ax.plot(t_plot, buff_v, marker=".", linestyle="None", markersize=2, label="Buffer Voltage (V)")
    ax.set_xlabel("Simulation Time (s)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("Energy System Voltages (RK4)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(out_dir, "voltages_rk4.pdf"), format="pdf")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="inputs/test.csv")
    ap.add_argument("--row_idx", type=int, default=None)
    ap.add_argument("--out_base", type=str, default="outputs/rk4")
    args = ap.parse_args()

    rows = read_rows(args.input)
    if len(rows) == 0:
        raise ValueError(f"No rows found in input CSV: {args.input}")

    if args.row_idx is None:
        indices = list(range(len(rows)))
    else:
        if args.row_idx < 0 or args.row_idx >= len(rows):
            raise ValueError(f"row_idx out of range: {args.row_idx} (rows={len(rows)})")
        indices = [args.row_idx]

    for row_idx in indices:
        params = parse_row(rows[row_idx])

        rtag = f"r{row_idx+1:02d}"
        out_dir = os.path.join(args.out_base, rtag)

        node, buff, states, exec_total_s, exec_per_step_s, n_steps, duty_pct, op = run_rk4(params)

        meta = {
            "method": "rk4",
            "row_idx": str(row_idx),
            "input_csv": args.input,

            "dt_s": f"{params['dt_s']:.12g}",
            "dur_s": f"{params['dur_s']:.12g}",
            "n_steps": str(n_steps),

            "exec_time_total_s": f"{exec_total_s:.9f}",
            "exec_time_per_step_s": f"{exec_per_step_s:.12e}",

            "duty_cycle_percent": f"{duty_pct:.6f}",
        }

        write_outputs(out_dir, node, buff, states, meta, params)

        write_opcount_csv(
            os.path.join(out_dir, "opcount.csv"),
            params["dt_s"], params["dur_s"],
            n_steps,
            op,
        )

        print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()