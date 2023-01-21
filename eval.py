import os
import sys
import pandas as pd
sys.path.append("src")
sys.path.append("src/models")
import torch.optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix as sklearn_cm
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def test(model, model_gt, dataloader, level=3):
    model.eval()

    logprobabilities = list()
    targets_list = list()
    gt_instance_list = list()
    logprobabilities_refined = list()
    #get the corresponding target gt of given level
    for iteration, data in tqdm(enumerate(dataloader)):
        if iteration==10:
            break
        if level==1:
            inputs, _, targets, _, gt_instance = data
        elif level ==2:
            inputs, _, _, targets, gt_instance = data
        else:
            inputs, targets, _, _, gt_instance = data

        del data

        if torch.cuda.is_available():
            inputs = inputs.cuda()

        
        y = targets.numpy()
        #y_i = gt_instance.cpu().detach().numpy()
        y_i = gt_instance.detach().numpy()

        z3, z1, z2 = model.forward(inputs)

        if model_gt is not None:
            #z3_refined = model_gt([z1.detach(), z2.detach(), z3.detach()])
            z3_refined = model_gt([z1, z2, z3])
        else:
            z3_refined = z3


        if type(z3_refined) == tuple:
            z3_refined = z3_refined[0]
            

        z1 = z1.detach().cpu().numpy()
        z2 = z2.detach().cpu().numpy()
        z3 = z3.detach().cpu().numpy()
        z3_refined = z3_refined.detach().cpu().numpy()
        
        targets_list.append(y)
        gt_instance_list.append(y_i)

        if level==1:
            logprobabilities.append(z1)
        elif level ==2:
            logprobabilities.append(z2)
        else:
            logprobabilities.append(z3)
        # NOTE the refined prediction is always based on level 3
        logprobabilities_refined.append(z3_refined)
        
        del z1, z2, z3, z3_refined


    return np.vstack(logprobabilities), np.concatenate(targets_list), np.vstack(gt_instance_list), np.vstack(logprobabilities_refined)

def confusion_matrix_to_accuraccies(confusion_matrix):

    confusion_matrix = confusion_matrix.astype(float)
    # sum(0) <- predicted sum(1) ground truth

    total = np.sum(confusion_matrix)
    n_classes, _ = confusion_matrix.shape
    overall_accuracy = np.sum(np.diag(confusion_matrix)) / total

    # calculate Cohen Kappa (https://en.wikipedia.org/wiki/Cohen%27s_kappa)
    N = total
    p0 = np.sum(np.diag(confusion_matrix)) / N
    pc = np.sum(np.sum(confusion_matrix, axis=0) * np.sum(confusion_matrix, axis=1)) / N ** 2
    kappa = (p0 - pc) / (1 - pc)

    recall = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=1) + 1e-12)
    precision = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=0) + 1e-12)
    f1 = (2 * precision * recall) / ((precision + recall) + 1e-12)

    # Per class accuracy
    cl_acc = np.diag(confusion_matrix) / (confusion_matrix.sum(1) + 1e-12)
    
    return overall_accuracy, kappa, precision, recall, f1, cl_acc

def build_confusion_matrix(targets, predictions):
    
    labels = np.unique(targets)
    labels = labels.tolist()
    #nclasses = len(labels)
        
    cm = sklearn_cm(targets, predictions, labels=labels)
#    precision = precision_score(targets, predictions, labels=labels, average='macro')
#    recall = recall_score(targets, predictions, labels=labels, average='macro')
#    f1 = f1_score(targets, predictions, labels=labels, average='macro')
#    kappa = cohen_kappa_score(targets, predictions, labels=labels)
    #print('precision, recall, f1, kappa: ', precision, recall, f1, kappa)
    
    return cm

def print_report(overall_accuracy, kappa, precision, recall, f1, cl_acc):
    
    report="""
    overall accuracy: \t{:.3f}
    kappa \t\t{:.3f}
    precision \t\t{:.3f}
    recall \t\t{:.3f}
    f1 \t\t\t{:.3f}
    """.format(overall_accuracy, kappa, precision.mean(), recall.mean(), f1.mean())

    print(report)
    #print('Per-class acc:', cl_acc)
    return cl_acc


def evaluate_fieldwise(model, model_gt, dataset, batchsize=1, workers=1, viz=False, prediction_dir = None, experiment_id = 0, fold_num=5, level=3,
                        ignore_undefined_classes=False):

    if prediction_dir != None:
        prediction_dir = f"{prediction_dir}_{experiment_id}"
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir)

    model.eval()

    if model_gt is not None:
        model_gt.eval()

    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batchsize, num_workers=workers, shuffle=False)

    logprobabilites, targets, gt_instance, logprobabilites_refined = test(model, model_gt, dataloader, level)
    # TODO TODO save the two probabilities for average mapping. prob map with max(logprob, 1). np.mean(5 prob distributions), np.sum. 
    #predictions = logprobabilites.argmax(1)
    predictions_refined = logprobabilites_refined.argmax(1)


    # one dimensional array after being flattened
    predictions = predictions.flatten()
    targets = targets.flatten()
    gt_instance = gt_instance.flatten()
    predictions_refined = predictions_refined.flatten()

    # Ignore unknown class class_id=0
    if viz:
        valid_crop_samples = targets != 9999999999
    elif level == 2 and ignore_undefined_classes: 
        valid_crop_samples = (targets != 0) * (targets != 7) * (targets != 9) * (targets != 12)
    elif level == 2: 
        # used for level 2. the meaning of 7,9,12: actually the same categories (maize).
        targets[(targets == 7)] = 12
        targets[(targets == 9)] = 12
        predictions[(predictions == 7)] = 12
        predictions[(predictions == 9)] = 12
        valid_crop_samples = (targets != 0) * (targets != 7) * (targets != 9)
    else:
        # this is the valid crop samples we are going to use (also for level 3)
        valid_crop_samples = targets != 0
    # note that GT might not be available when doing inference
    targets_wo_unknown = targets[valid_crop_samples]
    predictions_wo_unknown = predictions[valid_crop_samples]
    gt_instance_wo_unknown = gt_instance[valid_crop_samples]
    predictions_refined_wo_unknown = predictions_refined[valid_crop_samples]

    labels = np.unique(targets_wo_unknown)
    print('Num class: ', str(labels.shape[0]))
    # evaluation of pixel-wise prediction. for level 3,  we use the refined predictions 
    if level == 3:
        confusion_matrix = build_confusion_matrix(targets_wo_unknown, predictions_refined_wo_unknown)
    else:
        confusion_matrix = build_confusion_matrix(targets_wo_unknown, predictions_wo_unknown)
    print_report(*confusion_matrix_to_accuraccies(confusion_matrix))

    print('Evaluation is done pizel level!')

    # pred can be level 1, 2, 3
    prediction_wo_fieldwise = np.zeros_like(targets_wo_unknown)
    # pred_refined can only be level 3
    prediction_wo_fieldwise_refined = np.zeros_like(targets_wo_unknown)

    # target_field, prediction_field (for given level), prediction_level_field_refined (level 3) and target_field_instance_id can be saved as a csv and then be joint with the shapefile for visualizations
    num_field = np.unique(gt_instance_wo_unknown).shape[0]
    target_field = np.ones(num_field) * 8888
    prediction_field = np.ones(num_field) * 9999
    prediction_field_refined = np.ones(num_field)*9999
    target_field_instance_id = np.zeros(num_field)
    
    count = 0
    for i in np.unique(gt_instance_wo_unknown).tolist():
        field_indexes = gt_instance_wo_unknown == i

        pred = predictions_wo_unknown[field_indexes] 
        pred = np.bincount(pred)
        pred = np.argmax(pred)
        prediction_wo_fieldwise[field_indexes] = pred  
        prediction_field[count] = pred # for visual. pred depends on given level (for level3 it is not refined)
        
        # the following lines are for refined predictions at level 3
        pred = predictions_refined_wo_unknown[field_indexes]
        pred = np.bincount(pred)
        pred = np.argmax(pred)
        prediction_wo_fieldwise_refined[field_indexes] = pred
        prediction_field_refined = pred # for visual. level 3 refined pred. Final Result.

        target = targets_wo_unknown[field_indexes]
        target = np.bincount(target)
        target = np.argmax(target)
        target_field[count] = target # for visual
        
        target_field_instance_id[count] = i # for visual

        count += 1
    # evaluation of field-wise prediction
    if level == 3:
        confusion_matrix = build_confusion_matrix(targets_wo_unknown, prediction_wo_fieldwise_refined)
    else:
        confusion_matrix = build_confusion_matrix(targets_wo_unknown, prediction_wo_fieldwise)

    print_report(*confusion_matrix_to_accuraccies(confusion_matrix))
    # the fieldwise-refined excluding unknow classes prediction is the final prediction we are going to evaluate.
    pix_accuracy = np.sum(prediction_wo_fieldwise_refined==targets_wo_unknown) / prediction_wo_fieldwise_refined.shape[0] #refined prediction is only applied to level3

    save_path = os.path.join(prediction_dir, f"predictions_level_{level}")
    if viz: #used in test we want to save the csv and some other data for visualization purpose
        if level == 3:
            # for visualization, use prediction_per_field_refined and field_instance_id. 
            np.savez(save_path, level=level, logprobabilites = logprobabilites, logprobabilites_refined = logprobabilites_refined, gt = targets, gt_instance = gt_instance, cm=confusion_matrix,
            prediction_per_field = prediction_field, prediction_per_field_refined = prediction_field_refined, gt_per_field = target_field, field_instance_id = target_field_instance_id)
            # for visual. note that prediction_field_reined is always level 3
            # you can further aggregate the predictions after running 5 rounds of training and evaluations
            vis_data = {
                'target_field_instance_id': target_field_instance_id,
                'target_field': target_field, 
                'prediction_field': prediction_field, 
                'prediction_field_refined': prediction_field_refined
            }
            df = pd.DataFrame(vis_data, dtype='int32')
            df.to_csv(os.path.join(prediction_dir, f"visual_pred_level_{level}.csv"), index=False)
        else:
            np.savez(save_path, level=level, logprobabilites = logprobabilites, targets = targets, gt_instance = gt_instance, cm=confusion_matrix,
            prediction_per_field = prediction_field, gt_per_field = target_field, field_instance_id = target_field_instance_id)
        

    return pix_accuracy