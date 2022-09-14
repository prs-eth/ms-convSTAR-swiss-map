# for train
python3 -m pdb train.py \
        --data ZueriCrop.hdf5 --fold 1 --gt_path GT_labels_19_21_GP.csv --checkpoint_dir trained_models_debugging
# for test
# python3 -m pdb test.py \
#         --data ZueriCrop.hdf5 --fold 1 --snapshot trained_models/fold1.pth