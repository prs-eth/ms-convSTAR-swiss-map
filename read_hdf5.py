import numpy as np
import h5py

hf = h5py.File('/scratch2/tmehmet/swiss_crop/S2_Raw_L2A_CH_2021_hdf5_train.hdf5', 'r')
print(hf.keys())

hf.close()
