import numpy as np
import h5py

hf = h5py.File('/scratch2/tmehmet/swiss_crop/S2_Raw_L2A_CH_2021_hdf5_train.hdf5', 'r')
print(hf.keys())

#<KeysViewHDF5 ['data', 'gt', 'gt_canton', 'gt_instance']>

data = hf.get('data')
gt = hf.get('gt')
gt_instance = hf.get('gt_instance')
gt_canton = hf.get('gt_canton')
print(data.shape)
print(gt.shape)
print(gt_instance.shape)
print(gt_canton.shape)

num_samples = data.shape[0]
outDir = '/scratch2/tmehmet/swiss_crop_samples/'

for i in range(10)
    outfile = outDir + str(i) + '.npz'
    X = data[i,...]
    y = gt[i,:,:,0]
    yi = gt_instance[i,:,:,0]
    yc = gt_canton[i,:,:,0]
    
    np.savez(outfile, x=x, y=y, yi=yi, yc=yc)


hf.close()
