#!/usr/bin/env python3
import os
import re
import csv
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np

def metrics(y_hat: np.ndarray, y_ref: np.ndarray):
    err = y_hat - y_ref
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    max_err = float(np.max(np.abs(err)))
    return mae, rmse, max_err

def indices_on_grid(t_ref: np.ndarray, dt_other: float, tol: float = 1e-9):
    """
    Return indices i of t_ref such that t_ref[i] lies on the grid k*dt_other (k integer),
    within tolerance. Assumes t_ref starts at 0 or near 0.
    """
    if dt_other <= 0:
        return np.array([], dtype=int)

    k = np.rint(t_ref / dt_other)                 # nearest integer k
    t_snap = k * dt_other
    mask = np.isclose(t_ref, t_snap, rtol=0.0, atol=tol)
    return np.where(mask)[0]

DT_RE = re.compile(r"(?:^|/)dt_([0-9]*\.?[0-9]+)(?:/|$)")

# ---------- CSV reading helpers ----------
def read_two_col_csv(path: Path):
    """Reads a 2-column CSV with header; returns (t_list, y_list)."""
    t, y = [], []
    with path.open(newline="") as f:
        r = csv.reader(f)
        _ = next(r, None)  # header
        for row in r:
            if not row:
                continue
            t.append(float(row[0]))
            y.append(float(row[1]))
    return t, y

def extract_dt_from_path(p: Path):
    m = DT_RE.search(p.as_posix())
    if not m:
        return None
    return float(m.group(1))

def find_logs(base_dir: Path, which: str):
    """which: 'node' or 'buff'"""
    fname = "log-node-v.csv" if which == "node" else "log-buff-v.csv"
    logs = sorted(base_dir.rglob(fname))
    items = []
    for lp in logs:
        dt = extract_dt_from_path(lp)
        items.append((float("inf") if dt is None else dt, lp))
    items.sort(key=lambda x: x[0])
    return items

def downsample(x, y, stride: int):
    if stride <= 1:
        return x, y
    return x[::stride], y[::stride]

# ---------- MLP helpers (only used if MLP args provided) ----------
def read_config_row(input_csv: Path, row_idx: int):
    with input_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if row_idx < 0 or row_idx >= len(rows):
        raise ValueError(f"row_idx out of range: {row_idx} (rows={len(rows)})")

    r = rows[row_idx]

    def gf(key):
        if key not in r or r[key] == "":
            raise KeyError(f"Missing '{key}' in row {row_idx}")
        return float(r[key])

    # 9 features used in your viz_mlp.py data_cfg
    return [
        gf("sa_m2"),
        gf("eff"),
        gf("vmp"),
        gf("c_f"),
        gf("esr_ohm"),
        gf("q0_c"),
        gf("p_on_w"),
        gf("vhi"),
        gf("vlo"),
    ]

def mlp_from_json(json_dict):
    import torch.nn as nn
    in_features = json_dict["in_features"]
    layers = []
    for layer_cfg in json_dict["layers"]:
        cls = layer_cfg["class"]
        if cls == "Linear":
            out_features = layer_cfg["out_features"]
            layers.append(nn.Linear(in_features, out_features))
            in_features = out_features
        elif cls == "ReLU":
            layers.append(nn.ReLU())
        else:
            raise ValueError(f"Layer class not implemented: {cls}")
    return nn.Sequential(*layers)

def mlp_predict_times(mlp_cfg: Path, mlp_pth: Path, mlp_norm: Path, cfg9, times):
    import numpy as np
    import torch

    # load cfg
    with mlp_cfg.open("r") as f:
        jd = json.load(f)

    mlp = mlp_from_json(jd)
    state_dict = torch.load(mlp_pth, map_location="cpu")
    mlp.load_state_dict(state_dict)
    mlp.eval()

    norm = torch.load(mlp_norm, map_location="cpu")
    t_mean = norm["t_mean"]  # (10,)
    t_std  = norm["t_std"]   # (10,)
    v_mean = norm["v_mean"]  # (1,) or scalar-ish
    v_std  = norm["v_std"]   # (1,) or scalar-ish

    # build input matrix: [cfg9..., t]
    cfg9 = np.asarray(cfg9, dtype=float).reshape(1, 9)
    times = np.asarray(times, dtype=float).reshape(-1, 1)
    X = np.hstack([np.repeat(cfg9, repeats=times.shape[0], axis=0), times])  # (N,10)

    X_t = torch.tensor(X, dtype=torch.float32)
    Xn = (X_t - t_mean) / t_std

    with torch.no_grad():
        pred_norm = mlp(Xn)  # (N,1) typically

    pred = pred_norm * v_std + v_mean
    pred_np = pred.cpu().numpy().squeeze()
    return pred_np

# ---------- Main plotting ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, required=True,
                    help="Base sweep directory (e.g., outputs/rk1_dt_s_sweep/row_0)")
    ap.add_argument("--which", choices=["node", "buff", "both"], default="node",
                    help="Which logs to plot")

    ap.add_argument("--stride", type=int, default=1,
                    help="Downsample stride for plotting only (1 = no downsample)")
    ap.add_argument("--max_curves", type=int, default=0,
                    help="Limit number of RK1 curves (0 = no limit)")
    ap.add_argument("--out", type=str, default="dt_s_sweep_plot.pdf",
                    help="Output plot filename (pdf/png)")
    ap.add_argument("--title", type=str, default="RK1 dt_s sweep",
                    help="Plot title")

    # ---- Optional MLP overlay ----
    ap.add_argument("--mlp_cfg", type=str, default=None,
                    help="Path to mlp-cfg.json (optional; enables overlay)")
    ap.add_argument("--mlp_pth", type=str, default=None,
                    help="Path to mlp.pt weights (optional; enables overlay)")
    ap.add_argument("--mlp_norm", type=str, default=None,
                    help="Path to mlp-norm.pt (optional; enables overlay)")
    ap.add_argument("--input_csv", type=str, default=None,
                    help="rk_configs.csv path (required if using MLP overlay)")
    ap.add_argument("--row_idx", type=int, default=None,
                    help="Row index in rk_configs.csv (required if using MLP overlay)")

    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"base_dir not found: {base_dir}")

    use_mlp = (args.mlp_cfg is not None) or (args.mlp_pth is not None) or (args.mlp_norm is not None)
    if use_mlp:
        if not (args.mlp_cfg and args.mlp_pth and args.mlp_norm):
            raise ValueError("For MLP overlay, you must provide --mlp_cfg --mlp_pth --mlp_norm")
        if args.input_csv is None or args.row_idx is None:
            raise ValueError("For MLP overlay, you must also provide --input_csv and --row_idx")

        mlp_cfg = Path(args.mlp_cfg)
        mlp_pth = Path(args.mlp_pth)
        mlp_norm = Path(args.mlp_norm)
        input_csv = Path(args.input_csv)
        row_idx = int(args.row_idx)

        cfg9 = read_config_row(input_csv, row_idx)
    else:
        cfg9 = None

    def plot_kind(kind: str, out_path: Path):
        items = find_logs(base_dir, kind)

        # ---------------- Accuracy vs reference (NO interpolation) ----------------
        # Reference = first curve in items (smallest dt, because items is sorted by dt)
        ref_dt, ref_path = items[0]
        t_ref, y_ref = read_two_col_csv(ref_path)
        t_ref = np.asarray(t_ref, dtype=float)
        y_ref = np.asarray(y_ref, dtype=float)

        print("\n=== Accuracy vs reference RK1 (first dt_s) ===")
        print(f"Reference: dt={ref_dt:g}  path={ref_path}")

        # RK1 vs reference
        for j, (dt, path) in enumerate(items):
            t, y = read_two_col_csv(path)
            t = np.asarray(t, dtype=float)
            y = np.asarray(y, dtype=float)

            if j == 0:
                # reference itself
                mae = rmse = maxe = 0.0
                n = len(y_ref)
            else:
                # Compare only at times present in BOTH:
                # Use indices on reference grid that align to dt
                idx = indices_on_grid(t_ref, dt, tol=1e-9)

                # Additionally, guard that reference time doesn't exceed this run's final time
                # (sometimes dur rounding differs slightly)
                idx = idx[t_ref[idx] <= t[-1] + 1e-9]

                if idx.size == 0:
                    print(f"rk1 dt={dt:g}: no shared times with reference (skipping)")
                    continue

                # Because it's a uniform grid, y at those times in this run is simply y[k] with k = round(t/dt)
                k = np.rint(t_ref[idx] / dt).astype(int)
                y_hat = y[k]
                y_true = y_ref[idx]

                mae, rmse, maxe = metrics(y_hat, y_true)
                n = idx.size

            print(f"rk1 dt={dt:g}: N={n:6d}  MAE={mae:.6g}  RMSE={rmse:.6g}  MaxErr={maxe:.6g}")

        # MLP vs reference (only on node plot, same as your overlay behavior)
        if use_mlp and kind == "node":
            # Evaluate MLP on the reference times directly (same times)
            pred = mlp_predict_times(Path(args.mlp_cfg), Path(args.mlp_pth), Path(args.mlp_norm), cfg9, t_ref)
            pred = np.asarray(pred, dtype=float)

            mae, rmse, maxe = metrics(pred, y_ref)
            print(f"MLP (vs ref): N={len(y_ref):6d}  MAE={mae:.6g}  RMSE={rmse:.6g}  MaxErr={maxe:.6g}")
        print("=== End accuracy ===\n")


        if args.max_curves and args.max_curves > 0:
            items = items[:args.max_curves]
        if not items:
            raise RuntimeError(f"No logs found for '{kind}' under {base_dir}")

        fig, ax = plt.subplots(figsize=(9, 4.5))

        # Plot RK1 curves
        for j, (dt, path) in enumerate(items):
            t, v = read_two_col_csv(path)
            t, v = downsample(t, v, args.stride)

            label = f"rk1 dt={dt:g}" if dt != float("inf") else f"rk1 ({path.parent})"

            if j == 0:
                # FIRST sweep (smallest dt): different shape/appearance
                ax.plot(
                    t, v,
                    label=label + " (reference)",
                    marker="o",      # different shape
                    markersize=3,
                    linestyle="-",
                    linewidth=2.0,
                )
            else:
                # All other sweeps
                ax.plot(
                    t, v,
                    label=label,
                    linestyle="-",
                    linewidth=1.0,
                )

        # Overlay MLP curve (default: only on node voltage)
        if use_mlp and kind == "node":
            # Use the densest RK1 time grid for MLP (smallest dt => items[0])
            _, path0 = items[0]
            t_full, _ = read_two_col_csv(path0)
            t_plot = t_full[::max(args.stride, 1)]

            pred = mlp_predict_times(
                Path(args.mlp_cfg),
                Path(args.mlp_pth),
                Path(args.mlp_norm),
                cfg9,
                t_plot,
            )

            # MLP: different shape/appearance
            ax.plot(
                t_plot, pred,
                label="MLP",
                marker="s",      # different shape
                markersize=1,
                linestyle="--",
                linewidth=2.0,
            )

        ax.set_xlabel("Simulation Time (s)")
        ax.set_ylabel("Voltage (V)")
        ax.set_title(f"{args.title} ({'Node' if kind=='node' else 'Buffer'} Voltage)")
        ax.grid(True, alpha=0.3)

        # If you have many dt curves, the legend gets huge; keep it but compact.
        ax.legend(fontsize=7, ncol=2, frameon=True)

        plt.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Saved: {out_path}  (rk1_curves={len(items)})")

    out = Path(args.out)
    if args.which in ("node", "buff"):
        plot_kind(args.which, out)
    else:
        stem = out.stem
        suffix = out.suffix if out.suffix else ".pdf"
        plot_kind("node", out.with_name(f"{stem}_node{suffix}"))
        plot_kind("buff", out.with_name(f"{stem}_buff{suffix}"))

if __name__ == "__main__":
    main()