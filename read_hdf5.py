import os
import numpy as np
import h5py

hf = h5py.File('/scratch2/tmehmet/swiss_crop/S2_Raw_L2A_CH_2021_hdf5_train.hdf5', 'r')
print(hf.keys())

#<KeysViewHDF5 ['data', 'gt', 'gt_canton', 'gt_instance']> d

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
if  not os.path.exists(outDir):
    os.mkdir(outDir)

offset  = 33000

for i in range(offset,num_samples):
    if i%1000==0:
        print('sammples: ',  i)

    try:
        x = data[i,...]
        y = gt[i,:,:,0]
        yi = gt_instance[i,:,:,0]
        yc = gt_canton[i,:,:,0]
    
        outfile = outDir + str(i) + '.npz'
        np.savez(outfile, x=x, y=y, yi=yi, yc=yc)

    except:
        print('sample read error: ', i) 


hf.close()
