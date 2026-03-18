# sim_esys_cap_rk2.py
#
# Midpoint RK2 (buffer charge only),
# Usage:
#   python3 scripts/sim_esys_cap_rk2.py --row_idx 0
#
# Writes:
#   outputs/rk2/r01/
#     log-node-v.csv      (FULL)
#     log-buff-v.csv      (FULL)
#     log-states.csv
#     meta.csv
#     opcount.csv
#     voltages_rk2.pdf    (downsampled visualization only)
# ------------------------------------------------------------

import os, csv, math, time, argparse
import matplotlib.pyplot as plt

IRR_W_P_M2 = 1366.1
PLOT_DOWNSAMPLE = 15


def calc_solar_current(irr_w_m2, sa_m2, eff, vmp):
    # I_sun = (Irr * Area * eff) / Vmp
    return (irr_w_m2 * sa_m2 * eff) / vmp if vmp > 0.0 else 0.0

def calc_node_discr(q_c, c_f, i_a, esr_ohm, power_w):
    # (q/C + I*R)^2 - 4*P*R
    return (q_c / c_f + i_a * esr_ohm) ** 2 - 4.0 * power_w * esr_ohm

def calc_node_voltage(disc, q_c, c_f, i_a, esr_ohm):
    # v = 0.5 * (q/C + I*R + sqrt(disc))
    return 0.5 * ((q_c / c_f) + i_a * esr_ohm + math.sqrt(disc))


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


def write_opcount_csv(path, dt_s, dur_s, steps, opcounts):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method","dt_s","dur_s","steps","add","mul","div","sqrt","comp"])
        w.writerow([
            "rk2", f"{dt_s:.12g}", f"{dur_s:.12g}", int(steps),
            int(opcounts["add"]), int(opcounts["mul"]), int(opcounts["div"]),
            int(opcounts["sqrt"]), int(opcounts["comp"]),
        ])


def run_rk2(params, timing_only: bool = False):
    sa_m2=params["sa_m2"]; eff=params["eff"]; vmp=params["vmp"]
    c_f=params["c_f"]; esr_ohm=params["esr_ohm"]; q0_c=params["q0_c"]
    p_on_w=params["p_on_w"]; vhi=params["vhi"]; vlo=params["vlo"]
    dt_s=params["dt_s"]; dur_s=params["dur_s"]

    op = dict(add=0, mul=0, div=0, sqrt=0, comp=0)

    t_s = 0.0
    qt_c = q0_c
    p_mode_w = 0.0

    # initial i1
    op["comp"] += 1
    if vmp > 0.0:
        op["mul"] += 2; op["div"] += 1
    i1_a = calc_solar_current(IRR_W_P_M2, sa_m2, eff, vmp)

    # initial node_v
    op["div"] += 1; op["mul"] += 1; op["add"] += 1
    op["mul"] += 1; op["mul"] += 2; op["add"] += 1
    disc = calc_node_discr(qt_c, c_f, i1_a, esr_ohm, p_mode_w)

    op["comp"] += 1
    if disc < 0.0:
        p_mode_w = 0.0
        op["div"] += 1; op["mul"] += 1; op["add"] += 1
        op["mul"] += 1; op["mul"] += 2; op["add"] += 1
        disc = calc_node_discr(qt_c, c_f, i1_a, esr_ohm, p_mode_w)

    op["div"] += 1; op["mul"] += 1; op["add"] += 2; op["sqrt"] += 1; op["mul"] += 1
    node_v = calc_node_voltage(disc, qt_c, c_f, i1_a, esr_ohm)

    # Solar array cannot push current above Vmp
    if vmp <= node_v and i1_a > 0.0:
        i1_a = 0.0

    # Load cannot operate below Vlo
    if node_v <= vlo and p_mode_w != 0.0:
        p_mode_w = 0.0

    # logs
    log_states = [[t_s, "OFF"]]
    log_node_v = [[t_s, node_v]]
    log_buff_v = [[t_s, qt_c / c_f]]

    # timing + duty
    time_on_accum = 0.0
    t_start = time.perf_counter()
    n_steps = 0

    while t_s < dur_s:
        t_s += dt_s
        n_steps += 1

        # ---- RK2 integration: compute two slope evaluations ----
        # Stage 1 (k1): Calculate load current from previous step's node voltage
        i3_k1 = 0.0
        op["comp"] += 2
        if node_v > 0.0 and p_mode_w > 0.0:
            op["div"] += 1
            i3_k1 = p_mode_w / node_v

        # Stage 1 slope: k1 = i1 - i3_k1
        op["add"] += 1
        k1 = i1_a - i3_k1

        # Stage 1 intermediate charge: qt_mid = qt + 0.5*dt*k1
        op["mul"] += 2; op["add"] += 1
        qt_mid = qt_c + 0.5 * dt_s * k1
        op["comp"] += 1
        if qt_mid < 0.0:
            qt_mid = 0.0

        # Calculate node voltage at midpoint to get load current for stage 2
        op["div"] += 1; op["mul"] += 1; op["add"] += 1
        op["mul"] += 1; op["mul"] += 2; op["add"] += 1
        disc_mid = calc_node_discr(qt_mid, c_f, i1_a, esr_ohm, p_mode_w)

        # If discriminant is negative, power demand is too high 
        op["comp"] += 1
        p_mid = p_mode_w
        if disc_mid < 0.0:
            p_mid = 0.0
            op["div"] += 1; op["mul"] += 1; op["add"] += 1
            op["mul"] += 1; op["mul"] += 2; op["add"] += 1
            disc_mid = calc_node_discr(qt_mid, c_f, i1_a, esr_ohm, p_mid)

        # Calculate node voltage at midpoint
        op["div"] += 1; op["mul"] += 1; op["add"] += 2; op["sqrt"] += 1; op["mul"] += 1
        node_v_mid = calc_node_voltage(disc_mid, qt_mid, c_f, i1_a, esr_ohm)

        # Calculate load current at midpoint: i3 = P / V
        i3_k2 = 0.0
        op["comp"] += 2
        if node_v_mid > 0.0 and p_mid > 0.0:
            op["div"] += 1
            i3_k2 = p_mid / node_v_mid

        # Stage 2 slope: k2 = i1 - i3_k2
        op["add"] += 1
        k2 = i1_a - i3_k2

        # Update charge: qt = qt + dt*k2
        op["mul"] += 1; op["add"] += 1
        qt_c = qt_c + dt_s * k2
        op["comp"] += 1
        if qt_c < 0.0:
            qt_c = 0.0

        # ---- END-OF-STEP: Update system state after charge integration ----
        # Step 1: Calculate nominal solar current (before Vmp clamping)
        op["comp"] += 1
        if vmp > 0.0:
            op["mul"] += 2; op["div"] += 1
        i1_nom = calc_solar_current(IRR_W_P_M2, sa_m2, eff, vmp)

        # Step 2: Compute final node voltage with nominal current and current load power
        op["div"] += 1; op["mul"] += 1; op["add"] += 1
        op["mul"] += 1; op["mul"] += 2; op["add"] += 1
        disc = calc_node_discr(qt_c, c_f, i1_nom, esr_ohm, p_mode_w)

        op["comp"] += 1
        if disc < 0.0:
            p_mode_w = 0.0
            log_states.append([t_s, "power too high"])
            op["div"] += 1; op["mul"] += 1; op["add"] += 1
            op["mul"] += 1; op["mul"] += 2; op["add"] += 1
            disc = calc_node_discr(qt_c, c_f, i1_nom, esr_ohm, p_mode_w)

        op["div"] += 1; op["mul"] += 1; op["add"] += 2; op["sqrt"] += 1; op["mul"] += 1
        node_v = calc_node_voltage(disc, qt_c, c_f, i1_nom, esr_ohm)

        # Step 3: Apply threshold logic (VHI/VLO)
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

        # Step 4: Apply Vmp clamping
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

        # Step 5: Apply VLO threshold
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
        log_node_v, log_buff_v, log_states,
        exec_time_total_s, exec_time_per_step_s,
        n_steps, duty_cycle_percent, op
    )


#------------------------Plot---------------------------
def plot_csvs_to_pdf(csv_paths, out_pdf, param_text=None, downsample=PLOT_DOWNSAMPLE):
    fig = plt.figure(figsize=(8.5, 4.0))
    if param_text:
        fig.suptitle(param_text, fontsize=9, y=0.98)
    for csv_path in csv_paths:
        xs, ys = [], []
        with open(csv_path, newline="") as f:
            r = csv.DictReader(f)
            ycol = [c for c in r.fieldnames if c != "t_s"][0]
            for row in r:
                xs.append(float(row["t_s"]))
                ys.append(float(row[ycol]))
        xs = xs[::max(int(downsample), 1)]
        ys = ys[::max(int(downsample), 1)]
        plt.plot(xs, ys, marker=".", linestyle="None", markersize=2, label=ycol)

    plt.xlabel("Simulation Time (s)")
    plt.ylabel("Voltage (V)")
    plt.title("Energy System Voltages (RK2)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_pdf, format="pdf")
    plt.close()


# ---------------- Write outputs ----------------
def write_outputs(out_dir, log_node_v, log_buff_v, log_states, meta, params):
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
        for k, v in meta.items():
            w.writerow([k, v])

    param_text = (
        f"sa_m2={params['sa_m2']:.4g}, eff={params['eff']:.3g}, vmp={params['vmp']:.3g}, "
        f"c_f={params['c_f']:.3g}, esr_ohm={params['esr_ohm']:.3g}, q0_c={params['q0_c']:.3g}, "
        f"p_on_w={params['p_on_w']:.3g}, vhi={params['vhi']:.3g}, vlo={params['vlo']:.3g}, "
        f"dt_s={params['dt_s']:.3g}, dur_s={params['dur_s']:.3g}"
    )

    plot_csvs_to_pdf(
        [node_csv, buff_csv],
        os.path.join(out_dir, "voltages_rk2.pdf"),
        param_text=param_text,
        downsample=PLOT_DOWNSAMPLE
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="inputs/test.csv")
    ap.add_argument("--row_idx", type=int, default=None)
    ap.add_argument("--out_base", type=str, default="outputs/rk2")
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

        node, buff, states, exec_total_s, exec_per_step_s, n_steps, duty_pct, op = run_rk2(params)

        meta = {
            "method": "rk2/midpoint",
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
        write_opcount_csv(os.path.join(out_dir, "opcount.csv"), params["dt_s"], params["dur_s"], n_steps, op)

        print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()