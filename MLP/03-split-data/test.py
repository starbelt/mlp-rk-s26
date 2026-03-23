import os
import numpy as np

data_tst = np.load("./tst/000.npy")
data_trn = np.load("./trn/000.npy")
data_val = np.load("./val/000.npy")

print(f"Test Data Size: {len(data_tst)}") # Print Size of Test Data
print(f"Validation Data Size: {len(data_val)}") # Print Size of Test Data
print(f"Training Data Size: {len(data_trn)}") # Print Size of Test Data