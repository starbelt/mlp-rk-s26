# split_data.py
#
# Usage: python3 split_data.py /path/to/src/ /path/to/dst/
#  Reads all NPY files in src and splits into dst/trn, dst/val, and dst/tst
# Parameters:
#  /path/to/src/: a directory containing NPY files
#  /path/to/dst/: a directory containing trn, val, and tst directories
# Output:
#  A train, validate, and test split of the dataset

# import Python modules
import json        # json
import numpy as np # numpy
import os          # listdir
import shutil      # file copy
import sys         # argv

# "constants"

## train, validate, and test split (must add to 1.0)
TRN_FRAC = 0.6
VAL_FRAC = 0.2
#TST_FRAC = 0.2 # calculated as 1.0-(TRN_FRAC+VAL_FRAC)

# helper functions
## None

# initialize script arguments
src = '' # a directory containing NPY files
dst = '' # a directory containing trn, val, and tst directories

# parse script arguments
if len(sys.argv)==3:
  src = sys.argv[1]
  dst = sys.argv[2]
else:
  print(\
   'Usage: '\
   'python3 split_data.py /path/to/src/ /path/to/dst/'\
  )
  exit()

# trn, val, and tst directory paths
dst_trn = os.path.join(dst,'trn')
dst_val = os.path.join(dst,'val')
dst_tst = os.path.join(dst,'tst')

os.makedirs(dst_trn, exist_ok=True)
os.makedirs(dst_val, exist_ok=True)
os.makedirs(dst_tst, exist_ok=True)

# copy the npy-to-cfg key to trn, val, and tst
shutil.copy(os.path.join(src,'npy-to-cfg.json'), dst_trn)
shutil.copy(os.path.join(src,'npy-to-cfg.json'), dst_val)
shutil.copy(os.path.join(src,'npy-to-cfg.json'), dst_tst)

# collect NPY file paths
npys = [f for f in os.listdir(src) if f.endswith('.npy')]

# load data, split, and write
seed_i = 67
for npy in npys:
  # load data
  nparr = np.load(os.path.join(src,npy))
  num_samples = np.shape(nparr)[0]
  # split into trn, val, and tst
  rng = np.random.default_rng(seed=seed_i)
  shuffled_indices = rng.permutation(num_samples)
  trn_max = int(TRN_FRAC*num_samples)
  val_max = trn_max+int(VAL_FRAC*num_samples)
  tst_max = num_samples
  # trn
  trn_mask = np.zeros(num_samples, dtype=bool)
  trn_mask[shuffled_indices[0:trn_max]] = True
  trn_arr = nparr[trn_mask]
  np.save(os.path.join(dst_trn,npy),trn_arr)
  # val
  val_mask = np.zeros(num_samples, dtype=bool)
  val_mask[shuffled_indices[trn_max:val_max]] = True
  val_arr = nparr[val_mask]
  np.save(os.path.join(dst_val,npy),val_arr)
  # tst
  tst_mask = np.zeros(num_samples, dtype=bool)
  tst_mask[shuffled_indices[val_max:tst_max]] = True
  tst_arr = nparr[tst_mask]
  np.save(os.path.join(dst_tst,npy),tst_arr)
  # update seed
  seed_i = seed_i+1
