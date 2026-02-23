# gen_data.py
#
# Usage: python3 gen_data.py /path/to/cfg.json /path/to/logdir/
#  Reads a configuration JSON file and writes dataset NPY files to logdir
# Parameters:
#  /path/to/cfg.json: specifies amplitudes, frequencies, and phases in degrees
#  /path/to/logdir/: the path to the log directory for dataset files
# Output:
#  A dataset of NPY files

# import Python modules
import itertools   # product
import json        # JSON
import math        # ceil
import numpy as np # numpy
import os          # path.join
import sys         # argv
import csv         # CSV reader
from tqdm import tqdm

# "constants"
SA = 0 # index for surface area
EFF = 1 # index for efficiency
VMP = 2 # index for voltage at max power
CAP = 3 #index for capacitance 
ESR = 4 # index for equivalent series resistance
Q0 = 5 #index for initial charge
P = 6 #index for power
VHI = 7 # index for high voltage
VLO = 8 # index for low voltage
IRR_W_P_M2 = 1366.1


# helper functions

def calc_solar_current(irr_w_m2, sa_m2, eff, vmp):
    return (irr_w_m2 * sa_m2 * eff) / vmp if vmp > 0.0 else 0.0

def calc_node_discr(q_c, c_f, i_a, esr_ohm, power_w):
    return (q_c / c_f + i_a * esr_ohm) ** 2 - 4.0 * power_w * esr_ohm

def calc_node_voltage(disc, q_c, c_f, i_a, esr_ohm):
    return ((q_c / c_f) + i_a * esr_ohm + math.sqrt(disc)) / 2.0

# main simulation function
def run_truth(params):
    sa_m2=params["sa_m2"]; eff=params["eff"]; vmp=params["vmp"]
    c_f=params["c_f"]; esr_ohm=params["esr_ohm"]; q0_c=params["q0_c"]
    p_on_w=params["p_on_w"]; vhi=params["vhi"]; vlo=params["vlo"]
    dur_s=params["dur_s"]; dt_s = params["dt_s"]

    # initial values 
    t_s   = 0.0
    i1_a  = calc_solar_current(IRR_W_P_M2, sa_m2, eff, vmp)
    qt_c  = q0_c
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

    log_node_v = [[t_s, node_v]]
    log_buff_v = [[t_s, qt_c / c_f]]
    log_states = [[t_s, "OFF"]]

    #t_start = time.perf_counter()

    # loop structure
    while log_node_v[-1][0] < dur_s:
        t_s += dt_s

        i3_a = p_mode_w / node_v

        qt_c += (i1_a - i3_a) * dt_s
        if qt_c < 0.0:
            qt_c = 0.0

        i1_a = calc_solar_current(IRR_W_P_M2, sa_m2, eff, vmp)

        if p_mode_w == 0.0 and node_v >= vhi:
            p_mode_w = p_on_w
            log_states.append([t_s, "VHI"])

        node_discr = calc_node_discr(qt_c, c_f, i1_a, esr_ohm, p_mode_w)
        if node_discr < 0.0:
            p_mode_w = 0.0
            node_discr = calc_node_discr(qt_c, c_f, i1_a, esr_ohm, p_mode_w)
            log_states.append([t_s, "power too high"])

        node_v = calc_node_voltage(node_discr, qt_c, c_f, i1_a, esr_ohm)

        if vmp <= node_v and i1_a > 0.0:
            i1_a = 0.0
            log_states.append([t_s, "node voltage too high"])

        if node_v <= vlo and p_mode_w != 0.0:
            p_mode_w = 0.0
            log_states.append([t_s, "VLO"])

        log_node_v.append([t_s, node_v])
        log_buff_v.append([t_s, qt_c / c_f])

    #exec_time_s = time.perf_counter() - t_start
    return log_node_v

# initialize script arguments
cfg = '' # path to configuration file
log = '' # path to log file

# parse script arguments
if len(sys.argv)==3:
  cfg = sys.argv[1]
  log = sys.argv[2]
else:
  print(\
   'Usage: '\
   'python3 gen_data.py /path/to/cfg.json /path/to/logdir/'\
  )
  exit()

# load configuration parameters

exclude = {"dt_s", "dur_s"}
data = {}
meta = {}

with open(cfg, newline = "") as ifile:
  reader = csv.DictReader(ifile)
  for row in reader:
    for key, value in row.items():
      if key in exclude:
        meta.setdefault(key, []).append(float(value))
      else:
        data.setdefault(key, []).append(float(value))

json_file = "data-cfg.json"
#json_file = os.path.join("02-gen-data", "data-cfg.json")

with open(json_file, "w") as f:
    json.dump(data, f, indent=2)


# determine zfill padding
n = len(data["sa_m2"])  # number of rows / cases
#n = 2  # number of rows / cases
dt_s_avg = 0.1 #mean time step to secure the same number of points.
pad = max(1, math.floor(math.log10(n)) + 1)

for k in meta:
    meta[k] = meta[k][:n]

max_dur_s = max(meta["dur_s"])
max_points = int(max_dur_s/dt_s_avg)
#print(max_points)

# Ensures the same number of points per data set
for i in range(len(meta["dt_s"])):
    meta["dt_s"][i] = meta["dur_s"][i]/max_points
    #print(meta["dur_s"][i]/meta["dt_s"][i])

# for each wave, write out a dataset 
id_to_cfg = {}
cap_id = 0
for cap_id in tqdm(range(n), desc="Generating Data"):
  cap_id_str = str(cap_id).zfill(pad)
  super_cap = [
      data["sa_m2"][cap_id],   # SA
      data["eff"][cap_id],     # EFF
      data["vmp"][cap_id],     # VMP
      data["c_f"][cap_id],     # CAP
      data["esr_ohm"][cap_id], # ESR
      data["q0_c"][cap_id],    # Q0
      data["p_on_w"][cap_id],  # P
      data["vhi"][cap_id],     # VHI
      data["vlo"][cap_id],     # VLO
  ]
  sa = super_cap[SA]
  eff = super_cap[EFF]
  vmp = super_cap[VMP]
  c = super_cap[CAP]
  esr = super_cap[ESR]
  q0 = super_cap[Q0]
  p = super_cap[P]
  vhi = super_cap[VHI]
  vlo = super_cap[VLO]

  # derive wave parameters

  id_to_cfg[cap_id_str] = {
   'surface area': sa,
   'efficiency': eff,
   'max power voltage': vmp,
   'capacitance': c,
   'equivalent series resistance': esr,
   'initial charge': q0,
   'power': p,
   'high voltage': vhi,
   'low voltage': vlo
  }

  params = {
   'sa_m2': sa,
   'eff': eff,
   'vmp': vmp,
   'c_f': c,
   'esr_ohm': esr,
   'q0_c': q0,
   'p_on_w': p,
   'vhi': vhi,
   'vlo': vlo,
   'dur_s': meta["dur_s"][cap_id],
   'dt_s': meta["dt_s"][cap_id]
  }

  cap_out = np.array(run_truth(params), dtype=float)
  np.save(os.path.join(log,cap_id_str+'.npy'),cap_out)

# write a JSON configuration key
with open(os.path.join(log,'npy-to-cfg.json'), 'w') as ofile:
  json.dump(id_to_cfg,ofile)
