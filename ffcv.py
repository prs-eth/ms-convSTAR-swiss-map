import torch
import numpy as np
import torch.utils.data
import os, glob, json, csv
import h5py
from ffcv.fields import NDArrayField, FloatField, IntField


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, t=0.9, mode='all', eval_mode=False, fold=None, gt_path='labelsC.csv',
                 time_downsample_factor=2, num_channel=4, apply_cloud_masking=False, cloud_threshold=0.1,
                 return_cloud_cover=False, small_train_set_mode=False, data_canton_labels_dir=None, canton_ids_train=None):
        
        self.data = h5py.File(path, "r", libver='latest', swmr=True)
        self.samples = self.data["data"].shape[0]
        self.max_obs = self.data["data"].shape[1]
        self.spatial = self.data["data"].shape[2:-1]
        self.t = t
        self.augment_rate = 0.66
        self.eval_mode = eval_mode
        self.fold = fold
        self.num_channel = num_channel
        self.apply_cloud_masking = apply_cloud_masking
        self.cloud_threshold = cloud_threshold
        self.return_cloud_cover = return_cloud_cover
        self.data_canton_labels = json.load(open(data_canton_labels_dir))
        self.canton_ids_train = canton_ids_train

        # return the patch indices depending on the mode "train" or "test"
        self.valid_list = self.get_valid_list(mode)
        self.valid_samples = self.valid_list.shape[0]

        gt_path_ = './utils/' + gt_path        
        if not os.path.exists(gt_path_):
            gt_path_ = './'  + gt_path        
        
        file=open(gt_path_, "r")
        tier_1 = []
        tier_2 = []
        tier_3 = []
        tier_4 = []
        tier_code = []
        reader = csv.reader(file)
        for line in reader:
            tier_1.append(line[-5]) #'1st_tier'
            tier_2.append(line[-4]) #'2nd_tier'
            tier_3.append(line[-3]) #'3rd_tier'
            tier_4.append(line[-2]) #'4th_tier_ENG'
            tier_code.append(line[1]) #'LNF_code'

        tier_2[0] = '0_unknown'
        tier_3[0] = '0_unknown'
        tier_4[0] = '0_unknown'
    
        self.label_list = {}
        for i in range(len(tier_2)):
            if tier_1[i] == 'Vegetation' and tier_4[i] != '':
                # the mapping between numerical indices and LNF_code
                self.label_list[i] = int(tier_code[i])

            if tier_2[i] == '':
                tier_2[i] = '0_unknown'
            if tier_3[i] == '':
                tier_3[i] = '0_unknown'
            if tier_4[i] == '':
                tier_4[i] = '0_unknown'
            
        tier_2_elements = list(set(tier_2)) # len of list 6
        tier_3_elements = list(set(tier_3)) # 20
        tier_4_elements = list(set(tier_4)) # 52
        tier_2_elements.sort()
        tier_3_elements.sort()
        tier_4_elements.sort()
        
        # to map the predicted indices back to names, use tier_4_elements[index]
        tier_2_ = []
        tier_3_ = []
        tier_4_ = []
        for i in range(len(tier_2)):
            tier_2_.append(tier_2_elements.index(tier_2[i]))
            tier_3_.append(tier_3_elements.index(tier_3[i]))
            tier_4_.append(tier_4_elements.index(tier_4[i]))        

        self.label_list_local_1 = []
        self.label_list_local_2 = []
        self.label_list_glob = []
        self.label_list_local_1_name = []
        self.label_list_local_2_name = []
        self.label_list_glob_name = []
        for gt in self.label_list.keys(): # gt are only ids of rows that have tier_1 as 'vegetation' and tier 4 not none
            self.label_list_local_1.append(tier_2_[int(gt)])
            self.label_list_local_2.append(tier_3_[int(gt)])
            self.label_list_glob.append(tier_4_[int(gt)])
            
            self.label_list_local_1_name.append(tier_2[int(gt)])
            self.label_list_local_2_name.append(tier_3[int(gt)])
            self.label_list_glob_name.append(tier_4[int(gt)])

        # +1 represents the 'unknown' class. the actual n_classes contained in self.label_list is only 48, 52 is the number of all original classes
        self.n_classes = max(self.label_list_glob) + 1 #52
        self.n_classes_local_1 = max(self.label_list_local_1) + 1 #6
        self.n_classes_local_2 = max(self.label_list_local_2) + 1 #20

        print('Dataset size: ', self.samples)
        print('Valid dataset size: ', self.valid_samples)
        print('Sequence length: ', self.max_obs)
        print('Spatial size: ', self.spatial)
        print('Number of classes: ', self.n_classes)
        print('Number of classes - local-1: ', self.n_classes_local_1)
        print('Number of classes - local-2: ', self.n_classes_local_2)

        #for consistency loss---------------------------------------------------------
        self.l1_2_g = np.zeros(self.n_classes)
        self.l2_2_g = np.zeros(self.n_classes)
        self.l1_2_l2 = np.zeros(self.n_classes_local_2)
        
        # label_list_glob (or label_list_l3) is the mapping of label_list (selected elements of column 'GT') to hier4 labels
        for i in range(1,self.n_classes):
            if i in self.label_list_glob:
                self.l1_2_g[i] = self.label_list_local_1[self.label_list_glob.index(i)]
                self.l2_2_g[i] = self.label_list_local_2[self.label_list_glob.index(i)]
        # if the class is not in label_list, then the corresponding l1 mapping here is 0 (parent class 'unknown')
        for i in range(1,self.n_classes_local_2):
            if i in self.label_list_local_2:
                self.l1_2_l2[i] = self.label_list_local_1[self.label_list_local_2.index(i)]




    def __len__(self):
        return self.valid_samples


    def __getitem__(self, idx):
        # TODO save the hdf5 file in batch as .npz file to load the data faster
        idx = self.valid_list[idx]
        X = self.data["data"][idx]

        target_ = self.data["gt"][idx,...,0]
        gt_instance = self.data["gt_instance"][idx,...,0]

        X = np.transpose(X, (0, 3, 1, 2))

        #Change labels 
        target = np.zeros_like(target_)
        target_local_1 = np.zeros_like(target_)
        target_local_2 = np.zeros_like(target_)
        
        #here only the classes in label_list (Vegetation and hier4 is not none) get mapped. Other classes in target_ including no-data value 9999999 are not mapped (corresponding value in target is 0)
        #use the inversed mapping to map the predictions back to code
        for i, code in enumerate(list(self.label_list.values())):  
            target[target_ == code] = self.label_list_glob[i]
            target_local_1[target_ == code] = self.label_list_local_1[i]
            target_local_2[target_ == code] = self.label_list_local_2[i]
        
        # X = torch.from_numpy(X)
        # target = torch.from_numpy(target).float()
        # target_local_1 = torch.from_numpy(target_local_1).float()
        # target_local_2 = torch.from_numpy(target_local_2).float()
        # gt_instance = torch.from_numpy(gt_instance).float()

        #keep values between 0-1
        # X = X * 1e-4
        #Previous line should be modified as X = X / 4095 but not tested yet!
        X = X/4095

        return X.float(), target.long(), target_local_1.long(), target_local_2.long(), gt_instance.long()



    def get_valid_list(self, mode):
        valid = []
        if mode == "train":
            for k in self.canton_ids_train:
                valid += self.data_canton_labels[k]
        elif mode == "test":
            for k in self.data_canton_labels.keys():
                if k not in self.canton_ids_train:
                    valid += self.data_canton_labels[k]

        return np.array(valid)


    def get_rid_small_fg_tiles(self):
        valid = np.ones(self.samples)
        w,h = self.data["gt"][0,...,0].shape
        for i in range(self.samples):
            #if proportion of pixels in 24*24 patch is less than t, then this sample is marked as 0
            if np.sum( self.data["gt"][i,...,0] != 0 )/(w*h) < self.t:
                valid[i] = 0
        #return the indices of samples marked as 1 in valid (binary array)
        return np.nonzero(valid)[0]
        
    def split(self, mode):
        valid = np.zeros(self.samples)
        if mode=='test':
            valid[int(self.samples*0.75):] = 1.
        elif mode=='train':
            valid[:int(self.samples*0.75)] = 1.
        else:
            valid[:] = 1.

        w,h = self.data["gt"][0,...,0].shape
        for i in range(self.samples):
            if np.sum( self.data["gt"][i,...,0] != 0 )/(w*h) < self.t:
                valid[i] = 0
        
        return np.nonzero(valid)[0]

    








if __name__ == "__main__":

    mode = 'test'
    
    
    #datadir = '/cluster/work/igp_psr/tmehmet/S2_Raw_L2A_CH_2021_hdf5_train.hdf5'
    datadir = '/scratch2/tmehmet/swiss_crop/S2_Raw_L2A_CH_2021_hdf5_train.hdf5'

                
    #data_canton_labels_dir = "/cluster/work/igp_psr/tmehmet/S2_Raw_L2A_CH_2021_hdf5_train_canton_labels.json"
    data_canton_labels_dir = "/scratch2/tmehmet/swiss_crop/S2_Raw_L2A_CH_2021_hdf5_train_canton_labels.json"

    gt_path = 'GT_labels_19_21_GP.csv'
    canton_ids_train = ["0", "3", "5", "14", "18", "19", "20", "25"]

    if mode == 'train':
        hdf_dataset = Dataset(datadir, 0., 'train', False, gt_path, num_channel=4, apply_cloud_masking=False, data_canton_labels_dir=data_canton_labels_dir, canton_ids_train=canton_ids_train)
        write_path =  '/scratch2/tmehmet/swiss_crop/swiss_crop_train.beton'

    else:
        hdf_dataset = Dataset(datadir, 0., 'test', True, gt_path, num_channel=4, apply_cloud_masking=False, data_canton_labels_dir=data_canton_labels_dir, canton_ids_train=canton_ids_train)
        write_path =  '/scratch2/tmehmet/swiss_crop/debug_swiss_crop_test.beton'



    X, t, tl1, tl2, gti = hdf_dataset[0]
    print(X.shape)
    print(t.shape)
    print(tl2.shape)
    print(gti.shape)


    
    writer = DatasetWriter(write_path, {
        'X': NDArrayField(shape=(X.shape[0],X.shape[1],X.shape[2],X.shape[3],), dtype=np.dtype('float32')),
        'target': NDArrayField(t.shape[0],t.shape[1],),
        'target_local_1': NDArrayField(t.shape[0],t.shape[1],), dtype=np.dtype('long')),
        'target_local_2': NDArrayField(t.shape[0],t.shape[1],), dtype=np.dtype('long')),
        'gt_instance': NDArrayField(t.shape[0],t.shape[1],), dtype=np.dtype('long')),   

    }, num_workers=2)


    writer.from_indexed_dataset(hdf_dataset)