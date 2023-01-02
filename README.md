# ms-convSTAR
Pytorch implementation for hierarchical time series classification with multi-stage convolutional RNN described in: 

[Crop mapping from image time series: deep learning with multi-scale label hierarchies. Turkoglu, Mehmet Ozgur and D'Aronco, Stefano and Perich, Gregor and Liebisch, Frank and Streit, Constantin and Schindler, Konrad and Wegner, Jan Dirk. Remote Sensing of Environment, 2021.](https://arxiv.org/pdf/2102.08820.pdf)


## [[Paper]](https://arxiv.org/pdf/2102.08820.pdf)  - [[Poster]](https://drive.google.com/file/d/1UkzijujTMTFv-fwTs4cFjFIRQlJQoUrq/view?usp=sharing)


<img src="https://github.com/0zgur0/ms-convSTAR/blob/master/imgs/model_drawing.png">


If you find our work useful in your research, please consider citing our paper:

```bash
@article{turkoglu2021msconvstar,
  title={Crop mapping from image time series: deep learning with multi-scale label hierarchies},
  author={Turkoglu, Mehmet Ozgur and D'Aronco, Stefano and Perich, Gregor and Liebisch, Frank and Streit, Constantin and Schindler, Konrad and Wegner, Jan Dirk},
  journal={Remote Sensing of Environment},
  volume={264},
  year={2021},
  publisher={Elsevier}
}
```
## Getting Started
- All the data and scripts are stored in pf-pc04/scratch03/yihshe. Select the configued environment ms-convSTAR to run the code.
- The scripts train.py, dataset.py and eval.py have been annotated. Please check the inline annotations for more details.
- To run the training and test, use the shell file run.sh.

## Training
Please see line 16 to 26 of train.py for annotations of important variables and their current path. 
wandb is used to for training plots. Please modify it to suit your needs (line 303 to 308):
```bash
wandb.init(
    project='ms_convSTAR_CH_2021',
    entity='yihshe',
    name=f'experiment_{args.experiment_id}',
    config=args
)
```

Use run.sh to run the experiments for five rounds. The args have been set according to the current data paths e.g. for experiment 0, use
```bash
python3 -m train --seed 0 --experiment_id 0
```

TO-DO: Currently, the training is very slow due to the dataloading of hdf5 file. Some works need to be done to fix the slow dataloading in dataset.py e.g. save the hdf5 file in batch as npz files and then load them again (see line 131 of dataset.py)

## Testing 
Run the following comman in run.sh to test the model
```bash
python3 -m test --snapshot path/to/model --experiment_id 0 
```
Specify the experiment_id same as the arg for training so that you can find the correponding evaluations. By defult, the test.py will run evaluation for all three levels (line 83 to 85 of test.py). You can run the training and test for five times and then aggregate the results (npz files). For the bottom level, you can also aggregate the csv files and then use it for visualization (line 219 to 232 in eval.py). 

## Evaluation and Visualization
For visualization, the per-field predictions are saved as a csv file besides the npz file (line 225 to 232 of eval.py). The csv file has the following fields: 
  * 'target_field_instance_id': id of each target field
  * 'target_field': gt of the field
  * 'prediction_field': original prediction of each field
  * 'prediction_field_refined': refined prediction of each field (only for the bottom level)
Note that the predictions of the model are continuous ids. You will need to map them back to corresponding species names to have a meaningful predictions. See line 89 to 97 and line 151 to 154 of dataset.py. For example, at the bottom level, the predicted id is the element of the list self.label_list_glob; just find the element of the same index  of the list self.label_list_glob_name for the inverse mapping. For level 1, use self.label_list_local_1 and self.label_list_local_1_name; for level 2, use self.label_list_local_2 and self.label_list_local_2_name.
You can then join the csv and the shapefile LWB_vector_merged_processed for visualization. Use 'target_field_instance_id' of the csv and 'GLOBAL_ID' of the shapefile as the fields to join the two tables.




