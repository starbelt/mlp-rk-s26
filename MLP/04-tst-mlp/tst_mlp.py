# tst_mlp.py
#
# Usage:
#   python3 tst_mlp.py /path/to/mlp-cfg.json /path/to/mlp.pt /path/to/norm.pt /path/to/src/ /path/to/dst/
#
# Generates MLP defined by mlp-cfg.json, loads weights from mlp.pt, tests on
# every .npy file inside src/tst, and writes a summary CSV with columns:
#   configuration, mse_norm, rmse_volts, time_per_prediction
#
# Output:
#   - summary CSV across all test configurations

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

# "constants"
NUM_CPUS = os.cpu_count()


# helper functions
def mlp_from_json(json_dict):
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
            print('Layer class not yet implemented: ' + layer_class)
            print('  Exiting')
            exit()
    return nn.Sequential(*layers)


def count_hidden_layers(json_dict):
    num_linear = sum(
        1 for layer in json_dict['layers']
        if layer['class'] == 'Linear'
    )
    return max(0, num_linear - 1)


# dataset for ONE npy file
class MSDataset(Dataset):
    def __init__(self, src_dir, npy_name):
        with open(os.path.join(src_dir, 'npy-to-cfg.json'), 'r') as ifile:
            npy_to_cfg_dict = json.load(ifile)

        cap_id = os.path.splitext(npy_name)[0]

        if cap_id not in npy_to_cfg_dict:
            raise KeyError(f"{cap_id} not found in npy-to-cfg.json")

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

        npy_path = os.path.join(src_dir, npy_name)
        nparr = np.load(npy_path)

        self.data = torch.tensor(
            np.column_stack((
                np.repeat([data_cfg], repeats=nparr.shape[0], axis=0),
                nparr[:, 0]
            )),
            dtype=torch.float32
        )

        self.labels = torch.tensor(nparr[:, [1]], dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label


def evaluate_one_file(mlp, criterion, t_mean, t_std, v_mean, v_std, tst_dir, npy_name):
    tst_dataset = MSDataset(src_dir=tst_dir, npy_name=npy_name)

    worker_count = NUM_CPUS
    tst_loader = DataLoader(
        dataset=tst_dataset,
        batch_size=1024,
        shuffle=False,
        num_workers=worker_count
    )

    total_inference_time = 0.0
    total_predictions = 0

    # more exact MSE over all elements
    total_squared_error = 0.0
    total_elements = 0

    with torch.no_grad():
        for inputs, true_out in tqdm(tst_loader, desc=f'Evaluating {npy_name}'):
            inputs_norm = (inputs - t_mean) / t_std
            true_norm   = (true_out - v_mean) / v_std

            start = time.perf_counter()
            pred_norm = mlp(inputs_norm)
            end = time.perf_counter()

            batch_time = end - start
            total_inference_time += batch_time
            total_predictions += inputs.shape[0]

            squared_error = torch.sum((pred_norm - true_norm) ** 2).item()
            total_squared_error += squared_error
            total_elements += true_norm.numel()

    mse_norm = total_squared_error / total_elements
    rmse_volts = (mse_norm ** 0.5) * float(v_std.item())
    time_per_prediction = total_inference_time / total_predictions if total_predictions > 0 else 0.0

    return mse_norm, rmse_volts, time_per_prediction, total_predictions


# initialize script arguments
cfg = ''
pth = ''
norm_pth = ''
src = ''
dst = ''

# parse script arguments
if len(sys.argv) == 6:
    cfg = sys.argv[1]
    pth = sys.argv[2]
    norm_pth = sys.argv[3]
    src = sys.argv[4]
    dst = sys.argv[5]
else:
    print(
        'Usage: '
        'python3 tst_mlp.py '
        '/path/to/mlp-cfg.json '
        '/path/to/mlp.pt '
        '/path/to/norm.pt '
        '/path/to/src/ '
        '/path/to/dst/'
    )
    exit()

os.makedirs(dst, exist_ok=True)

# load JSON configuration of MLP
with open(cfg, 'r') as ifile:
    json_dict = json.load(ifile)

# get MLP cfg file name
mlp_id = os.path.splitext(os.path.basename(cfg))[0]

# create specified MLP model
mlp = mlp_from_json(json_dict)

# load state dictionary
state_dict = torch.load(pth, map_location='cpu')
mlp.load_state_dict(state_dict)
mlp.eval()

# load normalization parameters
norm = torch.load(norm_pth, map_location='cpu')
t_mean = norm["t_mean"]
t_std  = norm["t_std"]
v_mean = norm["v_mean"]
v_std  = norm["v_std"]

criterion = nn.MSELoss()
n_hidden = count_hidden_layers(json_dict)

tst_dir = os.path.join(src, 'tst')

# get all test npy files
npy_files = sorted(
    f for f in os.listdir(tst_dir)
    if f.endswith('.npy') and f != 'meta.npy'
)

if len(npy_files) == 0:
    print(f'No .npy files found in {tst_dir}')
    exit()

summary_rows = []

for npy_name in npy_files:
    config_id = os.path.splitext(npy_name)[0]

    mse_norm, rmse_volts, time_per_prediction, total_predictions = evaluate_one_file(
        mlp=mlp,
        criterion=criterion,
        t_mean=t_mean,
        t_std=t_std,
        v_mean=v_mean,
        v_std=v_std,
        tst_dir=tst_dir,
        npy_name=npy_name
    )

    print(f"{config_id}: Test RMSE = {rmse_volts:.6f} V")
    print(f"{config_id}: Avg inference time per prediction = {time_per_prediction * 1e6:.3f} µs")

    summary_rows.append([
        config_id,
        f'{mse_norm:.12f}',
        f'{rmse_volts:.6f}',
        f'{time_per_prediction:.12e}'
    ])

# write summary CSV
summary_file = os.path.join(dst, f"{mlp_id}-tst-summary.csv")
with open(summary_file, mode='w', newline='') as ofile:
    csvwriter = csv.writer(ofile)
    csvwriter.writerow(['configuration', 'mse_norm', 'rmse_volts', 'time_per_prediction'])
    csvwriter.writerows(summary_rows)

print(f"Wrote summary CSV: {summary_file}")
print(f"Number of hidden layers: {n_hidden}")