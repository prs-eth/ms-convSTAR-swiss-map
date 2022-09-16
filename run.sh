# for train
# python3 -m pdb train.py \
#         --data ZueriCrop.hdf5 --fold 1 --gt_path GT_labels_19_21_GP.csv --checkpoint_dir trained_models_debugging
# for test
# python3 -m pdb test.py \
#         --data ZueriCrop.hdf5 --fold 1 --snapshot trained_models/fold1.pth

python3 -m train --seed 0 --experiment_id 0 
# python3 -m train --seed 1 --experiment_id 1
# python3 -m train --seed 2 --experiment_id 2
# python3 -m train --seed 3 --experiment_id 3
# python3 -m train --seed 4 --experiment_id 4

