# sim_esys_cap_rk1.py
#
# Usage:
#   python3 scripts/sim_esys_cap_rk1.py --row_idx 0
#
# Writes:
#   outputs/rk1/r01/
#     log-node-v.npy
#     log-buff-v.npy
#     log-states.npy
#     meta.npy
#     step-times.csv
#     voltages_rk1.pdf
# ------------------------------------------------------------

import os, csv, math, time, argparse
import matplotlib.pyplot as plt
import numpy as np

IRR_W_P_M2 = 1366.1
PLOT_DOWNSAMPLE = 3  # ONLY for PDF visualization; CSV stays full.


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


# ---------------- Step timing CSV ----------------
def write_step_times_csv(path, step_timing_log):
    with open(path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t_s", "cumulative_step_time_sec"])
        for t_s, cumulative_step_time_s in step_timing_log:
            writer.writerow([f"{t_s:.12e}", f"{cumulative_step_time_s:.12e}"])


# ---------------- Simulation (RK1 / Euler) ----------------
def run_rk1(params, timing_only: bool = False):
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

    # ---- initial conditions ----
    t_s = 0.0
    imp_a = calc_solar_current(IRR_W_P_M2, sa_m2, eff, vmp)
    i1_a = imp_a
    qt_c = q0_c
    p_mode_w = 0.0  # OFF initially

    node_discr = calc_node_discr(qt_c, c_f, i1_a, esr_ohm, p_mode_w)
    if node_discr < 0.0:
        p_mode_w = 0.0
        node_discr = calc_node_discr(qt_c, c_f, i1_a, esr_ohm, p_mode_w)

    node_v = calc_node_voltage(node_discr, qt_c, c_f, i1_a, esr_ohm)

    # Solar array cannot push current above Vmp
    if vmp <= node_v and i1_a > 0.0:
        i1_a = 0.0

    # Load cannot operate below Vlo
    if node_v <= vlo and p_mode_w != 0.0:
        p_mode_w = 0.0

    # ---- logs ----
    log_states = [[t_s, "OFF"]]
    log_node_v = [[t_s, node_v]]
    log_buff_v = [[t_s, qt_c / c_f]]

    # cumulative timestep timing log
    step_timing_log = []
    cumulative_step_time_s = 0.0

    # Timing instrumentation
    time_on_accum = 0.0
    t_start = time.perf_counter()
    n_steps = 0

    # ---- Simulation loop: Euler method (RK1) ----
    while t_s < dur_s:
        # start timing this timestep
        step_start = time.perf_counter()

        t_s += dt_s
        n_steps += 1

        # ---- RK1 integration: compute charge update ----
        # Stage 1: Calculate load current from previous step's node voltage
        i3_a = 0.0
        if node_v > 0.0 and p_mode_w > 0.0:
            i3_a = p_mode_w / node_v

        # Stage 1 slope: k1 = i1 - i3
        k1 = i1_a - i3_a

        # Update charge: qt = qt + dt*k1
        qt_c += dt_s * k1
        if qt_c < 0.0:
            qt_c = 0.0

        # ---- END-OF-STEP: Update system state after charge integration ----
        # Step 1: Calculate nominal solar current (before Vmp clamping)
        i1_nom = calc_solar_current(IRR_W_P_M2, sa_m2, eff, vmp)

        # Step 2: Compute final node voltage with nominal current and current load power
        disc = calc_node_discr(qt_c, c_f, i1_nom, esr_ohm, p_mode_w)

        if disc < 0.0:
            p_mode_w = 0.0
            log_states.append([t_s, "power too high"])
            disc = calc_node_discr(qt_c, c_f, i1_nom, esr_ohm, p_mode_w)

        node_v = calc_node_voltage(disc, qt_c, c_f, i1_nom, esr_ohm)

        # Step 3: Apply threshold logic (VHI/VLO) - only checked at end of step
        if p_mode_w == 0.0:
            if node_v >= vhi:
                p_mode_w = p_on_w
                log_states.append([t_s, "VHI"])
                disc_check = calc_node_discr(qt_c, c_f, i1_nom, esr_ohm, p_mode_w)
                if disc_check < 0.0:
                    p_mode_w = 0.0
                    log_states.append([t_s, "power too high"])

        # Step 4: Apply Vmp clamping
        if vmp <= node_v and i1_nom > 0.0:
            i1_a = 0.0
            log_states.append([t_s, "node voltage too high"])
            disc_check = calc_node_discr(qt_c, c_f, i1_a, esr_ohm, p_mode_w)
            if disc_check < 0.0:
                p_mode_w = 0.0
                log_states.append([t_s, "power too high"])
        else:
            i1_a = i1_nom

        # Step 5: Apply VLO threshold
        if p_mode_w > 0.0:
            if node_v <= vlo:
                p_mode_w = 0.0
                log_states.append([t_s, "VLO"])

        # duty
        if p_mode_w > 0.0:
            time_on_accum += dt_s

        # Voltage logs (ONLY when not timing_only)
        if not timing_only:
            log_node_v.append([t_s, node_v])
            log_buff_v.append([t_s, qt_c / c_f])

        # end timing this timestep
        step_end = time.perf_counter()
        step_time_s = step_end - step_start
        cumulative_step_time_s += step_time_s
        step_timing_log.append([t_s, cumulative_step_time_s])

    # End timing
    exec_time_total_s = time.perf_counter() - t_start
    exec_time_per_step_s = exec_time_total_s / max(n_steps, 1)

    # duty cycle
    duty_cycle_percent = 100.0 * time_on_accum / max(dur_s, 1e-18)
    duty_cycle_percent = float(max(0.0, min(100.0, duty_cycle_percent)))

    # If timing_only, keep just one final voltage sample so return shape stays usable
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
        step_timing_log,
    )


# ---------------- Write outputs ----------------
def write_outputs(out_dir, log_node_v, log_buff_v, log_states, meta_dict, params):
    os.makedirs(out_dir, exist_ok=True)

    node_npy = os.path.join(out_dir, "log-node-v.npy")
    buff_npy = os.path.join(out_dir, "log-buff-v.npy")
    states_npy = os.path.join(out_dir, "log-states.npy")
    meta_npy = os.path.join(out_dir, "meta.npy")

    np.save(node_npy, np.asarray(log_node_v, dtype=np.float64))
    np.save(buff_npy, np.asarray(log_buff_v, dtype=np.float64))

    states_arr = np.array(
        [(float(t), str(s)) for t, s in log_states],
        dtype=[("t_s", "f8"), ("state", "U32")]
    )
    np.save(states_npy, states_arr)

    np.save(meta_npy, meta_dict)

    dn = max(int(PLOT_DOWNSAMPLE), 1)
    node_ds = log_node_v[::dn]
    buff_ds = log_buff_v[::dn]
    t_plot = [e[0] for e in node_ds]
    node_v = [e[1] for e in node_ds]
    buff_v = [e[1] for e in buff_ds]

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
    ax.set_title("Energy System Voltages (RK1)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(out_dir, "voltages_rk1.pdf"), format="pdf")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="rk_configs.csv")
    ap.add_argument("--row_idx", type=int, default=0)
    ap.add_argument("--out_base", type=str, default="RK1")

    ap.add_argument("--dt_s", type=float, default=None,
                    help="Override dt_s from CSV (seconds). If omitted, uses CSV dt_s.")

    args = ap.parse_args()

    rows = read_rows(args.input)
    if args.row_idx < 0 or args.row_idx >= len(rows):
        raise ValueError(f"row_idx out of range: {args.row_idx} (rows={len(rows)})")

    params = parse_row(rows[args.row_idx])

    if args.dt_s is not None:
        if args.dt_s <= 0.0:
            raise ValueError(f"--dt_s must be > 0, got {args.dt_s}")
        params["dt_s"] = float(args.dt_s)

    rtag = f"r{args.row_idx+1:02d}"
    out_dir = os.path.join(args.out_base, rtag)

    node, buff, states, exec_total_s, exec_per_step_s, n_steps, duty_pct, step_timing_log = run_rk1(params)

    meta = {
        "method": "rk1",
        "row_idx": str(args.row_idx),
        "input_csv": args.input,
        "dt_s": f"{params['dt_s']:.12g}",
        "dur_s": f"{params['dur_s']:.12g}",
        "n_steps": str(n_steps),
        "exec_time_total_s": f"{exec_total_s:.9f}",
        "exec_time_per_step_s": f"{exec_per_step_s:.12e}",
        "duty_cycle_percent": f"{duty_pct:.6f}",
    }

    write_outputs(out_dir, node, buff, states, meta, params)

    write_step_times_csv(
        os.path.join(out_dir, "step-times.csv"),
        step_timing_log,
    )

    print(f"Saved: {out_dir}")

if __name__ == "__main__":
    main()