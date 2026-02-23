# tst_mlp.py
#
# Usage: python3 tst_mlp.py /path/to/mlp-cfg.json /path/to/mlp.pt /path/to/src/ /path/to/dst/
#  Generates MLP defined by mlp-cfg.json, loads weights from mlp.pt, tests on
#  src/tst, and writes the results to dst
# Parameters:
#  /path/to/mlp-cfg.json: a JSON specification of an MLP
#  /path/to/mlp.pt: a PyTorch model weights file
#  /path/to/src/: a directory containing the tst/ directory
#  /path/to/dst/: a directory to write the performance statistics
# Output:
#  MLP performance statistics

# import Python modules
import csv            # csv writer
import json           # json
import numpy as np    # numpy
import os             # listdir
import sys            # argv
import time
import torch          # PyTorch
import torch.nn as nn # Sequential, Linear, ReLU
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datetime import datetime

# "constants"
NUM_CPUS = os.cpu_count()

# helper functions

## accepts a JSON dictionary (the result of json.load(ifile)) as input
## returns a PyTorch nn.Sequential MLP as output
def mlp_from_json(json_dict):
  in_features = json_dict['in_features']
  layers = []
  for layer_cfg in json_dict['layers']:
    layer_class = layer_cfg['class']
    if layer_class=='Linear':
      out_features = layer_cfg['out_features']
      layers.append(nn.Linear(in_features,out_features))
      in_features = out_features
    elif layer_class=='ReLU':
      layers.append(nn.ReLU())
    else:
      print('Layer class not yet implemented: '+layer_class)
      print('  Exiting')
      exit()
  return nn.Sequential(*layers)

# helper classes
def count_hidden_layers(json_dict):
    num_linear = sum(
        1 for layer in json_dict['layers']
        if layer['class'] == 'Linear'
    )
    # assume last Linear is output layer
    return max(0, num_linear - 1)

## custom dataset
class MSDataset(Dataset):
  # src_dir contains npy-to-cfg.json and .npy files
  def __init__(self, src_dir):
    # read npy-to-cfg.json
    npy_to_cfg_dict = {}
    with open(os.path.join(src_dir,'npy-to-cfg.json'), 'r') as ifile:
      npy_to_cfg_dict = json.load(ifile)
    # iteratively read .npy files and add to data and label
    npys = [f for f in os.listdir(src_dir) if f.endswith('.npy')]
    for npy in npys:
      cap_id = npy[:-4]
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
      nparr = np.load(os.path.join(src_dir,npy))
      self.data = torch.tensor(np.column_stack((\
       np.repeat([data_cfg],repeats=nparr.shape[0],axis=0),nparr[:,0]\
      )),dtype=torch.float32)
      self.labels = torch.tensor(nparr[:,[1]],dtype=torch.float32)

  # src_dir contains npy-to-cfg.json and .npy files
  def __len__(self):
    return len(self.data)

  # src_dir contains npy-to-cfg.json and .npy files
  def __getitem__(self, idx):
    sample = self.data[idx]
    label = self.labels[idx]
    return sample, label

# initialize script arguments
cfg = '' # a JSON specification of an MLP
pth = '' # a PyTorch model weights file
src = '' # a directory containing the tst/ directory
dst = '' # a directory to write the performance statistics
nomr_pth = '' # a PyTorch file containing the normalization parameters

# parse script arguments
if len(sys.argv)==6:
  cfg = sys.argv[1]
  pth = sys.argv[2]
  norm_pth = sys.argv[3]
  src = sys.argv[4]
  dst = sys.argv[5]
else:
  print(\
   'Usage: '\
   'python3 tst_mlp.py /path/to/mlp-cfg.json /path/to/mlp.pt /path/to/src/ /path/to/dst/'\
  )
  exit()

# load JSON configuration of MLP
json_dict = {}
with open(cfg, 'r') as ifile:
  json_dict = json.load(ifile)

# get MLP cfg file name
mlp_id = os.path.splitext(os.path.basename(cfg))[0]

# create specified MLP model
mlp = mlp_from_json(json_dict)

# load state dictionary
state_dict = torch.load(pth)
mlp.load_state_dict(state_dict)
mlp.eval()

norm = torch.load(norm_pth, map_location='cpu')
t_mean = norm["t_mean"]
t_std  = norm["t_std"]
v_mean = norm["v_mean"]
v_std  = norm["v_std"]

# criterion: use a regression loss function, specifically MSE
criterion = nn.MSELoss()

total_inference_time = 0.0
total_predictions = 0

# load test dataset and construct test data loader
tst_dataset = MSDataset(src_dir=os.path.join(src,'tst'))
worker_count = NUM_CPUS # no need for more than 16 data loader workers (changed from min(NUM_CPUS, 16) to NUM_CPUS)
tst_loader = DataLoader(\
 dataset=tst_dataset, batch_size=1024, shuffle=False, num_workers=worker_count\
)

# evaluate model
tst_loss = 0.0
with torch.no_grad():
  for inputs, true_out in tqdm(tst_loader, desc='Evaluating on test data'):
    inputs_norm = (inputs - t_mean) / t_std
    true_norm   = (true_out - v_mean) / v_std

    start = time.perf_counter()
    pred_norm = mlp(inputs_norm)
    end = time.perf_counter()

    batch_time = end - start
    total_inference_time += batch_time
    total_predictions += inputs.shape[0]
    
    loss = criterion(pred_norm, true_norm)
    tst_loss += loss.item()

# print MSE for test dataset
mse_norm = tst_loss / len(tst_loader)
rmse_volts = (mse_norm ** 0.5) * float(v_std.item())

print(f"Test RMSE: {rmse_volts:.6f} V")

time_per_prediction = total_inference_time / total_predictions
print(f"Avg inference time per prediction: {time_per_prediction*1e6:.3f} µs")

n_hidden = count_hidden_layers(json_dict)

# write MSE to CSV file
with open(os.path.join(dst,f"{mlp_id}-tstmse.csv"),mode='w',newline='') as ofile:
  csvwriter = csv.writer(ofile)
  csvwriter.writerow(['metric', 'value'])
  csvwriter.writerow(['mse_norm', f'{mse_norm:.12f}'])
  csvwriter.writerow(['rmse_volts', f'{rmse_volts:.6f}'])
  csvwriter.writerow(['time_per_prediction_sec', f'{time_per_prediction:.12e}'])

