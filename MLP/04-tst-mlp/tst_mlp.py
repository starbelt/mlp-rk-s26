# ============================================================
# tst_mlp.py
# ------------------------------------------------------------
# PURPOSE:
#   - Load a trained MLP model
#   - Run inference on test .npy files
#   - Compute error metrics (MSE, RMSE)
#   - Measure inference time
#   - Save results to a summary CSV
#
# USAGE:
#   python3 tst_mlp.py <cfg.json> <model.pt> <norm.pt> <src/> <dst/>
# ============================================================


# =========================
# Imports
# =========================
import csv
import json
import numpy as np
import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# =========================
# Constants
# =========================
NUM_CPUS = os.cpu_count()
NUM_RUNS = 1


# ============================================================
# Model Utilities
# ============================================================

def mlp_from_json(json_dict):
    """
    Build an MLP model from JSON configuration.
    """
    in_features = json_dict['in_features']
    layers = []

    for layer_cfg in json_dict['layers']:
        layer_class = layer_cfg['class']

        if layer_class == 'Linear':
            out_features = layer_cfg['out_features']
            layers.append(nn.Linear(in_features, out_features))
            in_features = out_features

        elif layer_class == 'ReLU':
            layers.append(nn.ReLU())

        else:
            print('Layer class not yet implemented:', layer_class)
            exit()

    return nn.Sequential(*layers)


def count_hidden_layers(json_dict):
    """
    Count number of hidden layers (Linear layers minus output layer).
    """
    num_linear = sum(
        1 for layer in json_dict['layers']
        if layer['class'] == 'Linear'
    )
    return max(0, num_linear - 1)


# ============================================================
# Dataset Definition
# ============================================================

class MSDataset(Dataset):
    """
    Dataset for a single .npy file.

    Combines:
      - Configuration parameters (from JSON)
      - Time input (from npy)
    """

    def __init__(self, src_dir, npy_name):

        # Load mapping from npy file -> configuration
        with open(os.path.join(src_dir, 'npy-to-cfg.json'), 'r') as ifile:
            npy_to_cfg_dict = json.load(ifile)

        cap_id = os.path.splitext(npy_name)[0]

        if cap_id not in npy_to_cfg_dict:
            raise KeyError(f"{cap_id} not found in npy-to-cfg.json")

        # Extract configuration parameters
        data_cfg = np.array([
            npy_to_cfg_dict[cap_id]['surface area'],
            npy_to_cfg_dict[cap_id]['efficiency'],
            npy_to_cfg_dict[cap_id]['max power voltage'],
            npy_to_cfg_dict[cap_id]['capacitance'],
            npy_to_cfg_dict[cap_id]['equivalent series resistance'],
            npy_to_cfg_dict[cap_id]['initial charge'],
            npy_to_cfg_dict[cap_id]['power'],
            npy_to_cfg_dict[cap_id]['high voltage'],
            npy_to_cfg_dict[cap_id]['low voltage']
        ])

        # Load time + voltage data
        npy_path = os.path.join(src_dir, npy_name)
        nparr = np.load(npy_path)

        # Build input features: [config params + time]
        self.data = torch.tensor(
            np.column_stack((
                np.repeat([data_cfg], repeats=nparr.shape[0], axis=0),
                nparr[:, 0]  # time column
            )),
            dtype=torch.float32
        )

        # Labels: voltage
        self.labels = torch.tensor(nparr[:, [1]], dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# ============================================================
# Evaluation Function
# ============================================================

def evaluate_one_file(mlp, criterion, t_mean, t_std, v_mean, v_std, tst_dir, npy_name):
    """
    Run inference on a single npy file and compute:
      - MSE (normalized)
      - RMSE (volts)
      - Time per prediction
    """

    tst_dataset = MSDataset(src_dir=tst_dir, npy_name=npy_name)

    tst_loader = DataLoader(
        dataset=tst_dataset,
        batch_size=1024,
        shuffle=False,
        num_workers=NUM_CPUS
    )

    total_inference_time = 0.0
    total_predictions = 0
    total_squared_error = 0.0
    total_elements = 0

    with torch.no_grad():
        for inputs, true_out in tqdm(tst_loader, desc=f'Evaluating {npy_name}'):

            # Normalize inputs + outputs
            inputs_norm = (inputs - t_mean) / t_std
            true_norm   = (true_out - v_mean) / v_std

            # Measure inference time
            start = time.perf_counter()
            pred_norm = mlp(inputs_norm)
            end = time.perf_counter()

            total_inference_time += (end - start)
            total_predictions += inputs.shape[0]

            # Compute squared error
            squared_error = torch.sum((pred_norm - true_norm) ** 2).item()
            total_squared_error += squared_error
            total_elements += true_norm.numel()

    # Final metrics
    mse_norm = total_squared_error / total_elements
    rmse_volts = (mse_norm ** 0.5) * float(v_std.item())
    time_per_prediction = (
        total_inference_time / total_predictions
        if total_predictions > 0 else 0.0
    )

    return mse_norm, rmse_volts, time_per_prediction, total_predictions


# ============================================================
# Argument Parsing
# ============================================================

if len(sys.argv) != 6:
    print(
        "Usage: python3 tst_mlp.py "
        "<cfg.json> <model.pt> <norm.pt> <src/> <dst/>"
    )
    exit()

cfg, pth, norm_pth, src, dst = sys.argv[1:6]


# ============================================================
# Setup + Load Model
# ============================================================

os.makedirs(dst, exist_ok=True)

# Load model configuration
with open(cfg, 'r') as ifile:
    json_dict = json.load(ifile)

mlp_id = os.path.splitext(os.path.basename(cfg))[0]

# Build model
mlp = mlp_from_json(json_dict)

# Load weights
state_dict = torch.load(pth, map_location='cpu')
mlp.load_state_dict(state_dict)
mlp.eval()

# Load normalization
norm = torch.load(norm_pth, map_location='cpu')
t_mean, t_std = norm["t_mean"], norm["t_std"]
v_mean, v_std = norm["v_mean"], norm["v_std"]

criterion = nn.MSELoss()
n_hidden = count_hidden_layers(json_dict)


# ============================================================
# Load Test Files
# ============================================================

tst_dir = os.path.join(src, 'tst')

npy_files = sorted(
    f for f in os.listdir(tst_dir)
    if f.endswith('.npy') and f != 'meta.npy'
)

if not npy_files:
    print(f"No .npy files found in {tst_dir}")
    exit()



# =========== [ Evaluation Loop ] ===========

summary_rows = []

for npy_name in npy_files:

    config_id = os.path.splitext(npy_name)[0]

    mse_list = []
    rmse_list = []
    time_list = []

    # --- run multiple times ---
    for _ in range(NUM_RUNS):
        mse_norm, rmse_volts, time_per_prediction, _ = evaluate_one_file(
            mlp, criterion,
            t_mean, t_std,
            v_mean, v_std,
            tst_dir, npy_name
        )

        mse_list.append(mse_norm)
        rmse_list.append(rmse_volts)
        time_list.append(time_per_prediction)

    # ====================================================
    # Compute statistics 
    # ====================================================

    avg_mse  = np.mean(mse_list)
    avg_rmse = np.mean(rmse_list)
    avg_time = np.mean(time_list)

    min_time = np.min(time_list)
    max_time = np.max(time_list)
    std_time = np.std(time_list)

    # ====================================================
    # Print results
    # ====================================================

    print(f"{config_id}: Avg RMSE = {avg_rmse:.6f} V")
    print(f"{config_id}: Avg time = {avg_time * 1e6:.3f} µs")
    print(f"{config_id}: Min/Max = {min_time*1e6:.3f}/{max_time*1e6:.3f} µs")
    print(f"{config_id}: Std Dev = {std_time*1e6:.3f} µs")

    # ====================================================
    # Save to CSV
    # ====================================================

    summary_rows.append([
        config_id,
        f'{avg_mse:.12f}',
        f'{avg_rmse:.6f}',
        f'{avg_time:.12e}',
        f'{min_time:.12e}',
        f'{max_time:.12e}',
        f'{std_time:.12e}'
    ])