import numpy as np
import h5py

hf = h5py.File('/scratch2/tmehmet/swiss_crop/S2_Raw_L2A_CH_2021_hdf5_train.hdf5', 'r')
print(hf.keys())

#<KeysViewHDF5 ['data', 'gt', 'gt_canton', 'gt_instance']>

print(data.shape)
print(gt.shape)

hf.close()
