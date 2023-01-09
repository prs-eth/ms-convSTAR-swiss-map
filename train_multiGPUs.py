import torch
import numpy as np
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn
import argparse
import os
from dataset_npz import Dataset
from models.multi_stage_sequenceencoder import multistageSTARSequentialEncoder, multistageLSTMSequentialEncoder
from models.networkConvRef import model_2DConv
from eval import evaluate_fieldwise
import wandb
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    




def parse_args():
    """
    To highlight a few important variables:
        --data, hdf5 file, preprocessed data used for training and testing
        --gt_path, csv file, hierarchy of crop labels 
        --seed, int, change the random seed for another round of training, and then aggregate the predictions
        --data_canton_labels, json file of a dict, patch indices of the hdf5 grouped by canton ids
        --canton_ids_train, list, selected cantons for training; the rest will be used for testing
        --checkpoint_dir, str, path to save the trained models
        --prediction_dir, str, path to save the predictions
    All default values of these variables are those currently being used.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--data", type=str, default='/scratch2/tmehmet/swiss_crop/S2_Raw_L2A_CH_2021_hdf5_train.hdf5', help="path to dataset")
    parser.add_argument('-dn', "--npz_dir", type=str, default='/scratch2/tmehmet/swiss_crop_samples/', help="path to dataset npz files")
    parser.add_argument('-b', "--batchsize", default=32, type=int, help="batch size")
    parser.add_argument('-w', "--workers", default=12, type=int, help="number of dataset worker threads")
    parser.add_argument('-e', "--epochs", default=30, type=int, help="epochs to train")
    parser.add_argument('-l', "--learning_rate", default=0.001, type=float, help="learning rate")
    parser.add_argument('-s', "--snapshot", default=None,
                        type=str, help="load weights from snapshot")
    parser.add_argument(   '-c', "--checkpoint_dir", default='/scratch2/tmehmet/swiss_crop_model',
                        type=str,help="directory to save checkpoints")
    parser.add_argument('-wd', "--weight_decay", default=0.0001, type=float, help="weight_decay")
    parser.add_argument('-hd', "--hidden", default=64, type=int, help="hidden dim")
    parser.add_argument('-nl', "--layer", default=6, type=int, help="num layer")
    parser.add_argument('-lrs', "--lrSC", default=2, type=int, help="lrScheduler")
    parser.add_argument('-nm', "--name", default='swiss_map', type=str, help="name")
    parser.add_argument('-l1', "--lambda_1", default=0.1, type=float, help="lambda_1")
    parser.add_argument('-l2', "--lambda_2", default=0.3, type=float, help="lambda_2")
    parser.add_argument('-l3', "--lambda_3", default=0.6, type=float, help="lambda_3")
    parser.add_argument('-l_gt', "--lambda_gt", default=0.6, type=float, help="lambda_gt")
    parser.add_argument('-dr', "--dropout", default=0.5, type=float, help="dropout of CNN")
    parser.add_argument('-stg', "--stage", default=3, type=float, help="num stage")
    parser.add_argument('-cp', "--clip", default=5, type=float, help="grad clip")
    parser.add_argument('-sd', "--seed", default=0, type=int, help="random seed")
    parser.add_argument('-fd', "--fold", default=1, type=int, help="5 fold")
    parser.add_argument('-gt', "--gt_path", default='GT_labels_19_21_GP.csv', type=str, help="gt file path")
    parser.add_argument('-cell', "--cell", default='star', type=str, help="Cell type: main building block")
    parser.add_argument('-id', "--input_dim", default=4, type=int, help="Input channel size")
    parser.add_argument('-cm', "--apply_cm", default=False, type=bool, help="apply cloud masking")
    parser.add_argument('-pred', "--prediction_dir", default='predictions', type=str,help="directory to save predictions")
    parser.add_argument('-exp', "--experiment_id", default=0, type=int, help="times of running the experiment")
    parser.add_argument('--data_canton_labels', default = "/scratch2/tmehmet/swiss_crop/S2_Raw_L2A_CH_2021_hdf5_train_canton_labels.json", type = str, help="Canton labels for each patch in gt")
    parser.add_argument('--canton_ids_train', default = ["0", "3", "5", "14", "18", "19", "20", "25"], type=list, help="Canton ids to train")
    parser.add_argument('-wdb', "--wandb_enable", default=True, type=bool, help="wandb")
    parser.add_argument('-ev', "--eval", action='store_true', help="eval mode")

    return parser.parse_args()

class stepCount():
    def __init__(self, init_step=0):
        self.step = init_step
    def count(self):
        self.step += 1
    def reset(self):
        self.step = 0

def main(
        datadir=None,
        npz_dir=None,
        data_canton_labels_dir=None,
        canton_ids_train=None,
        batchsize=1,
        workers=12,
        epochs=1,
        lr=1e-3,
        snapshot=None,
        checkpoint_dir=None,
        experiment_id=None,
        prediction_dir=None,
        weight_decay=0.0000,
        name='debug',
        layer=6,
        hidden=64,
        lrS=1,
        lambda_1=1,
        lambda_2=1,
        lambda_3=1,
        lambda_gt=1,
        stage=3,
        clip=1,
        fold_num=None,
        gt_path=None,
        cell=None,
        dropout=None,
        input_dim=None,
        apply_cm=None,
        wandb_enable=False,
        eval_mode=False
):
    checkpoint_dir = f"{checkpoint_dir}_{experiment_id}"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    traindataset = Dataset(datadir, 0., 'train', False, fold_num, gt_path, num_channel=input_dim, apply_cloud_masking=apply_cm, data_canton_labels_dir=data_canton_labels_dir, canton_ids_train=canton_ids_train,
    npz_dir=npz_dir)
    testdataset = Dataset(datadir, 0., 'test', True, fold_num, gt_path, num_channel=input_dim, apply_cloud_masking=apply_cm, data_canton_labels_dir=data_canton_labels_dir, canton_ids_train=canton_ids_train,
    npz_dir=npz_dir)
    
    nclasses = traindataset.n_classes
    nclasses_local_1 = traindataset.n_classes_local_1
    nclasses_local_2 = traindataset.n_classes_local_2

    LOSS_WEIGHT = torch.ones(nclasses)
    LOSS_WEIGHT[0] = 0
    LOSS_WEIGHT_LOCAL_1 = torch.ones(nclasses_local_1)
    LOSS_WEIGHT_LOCAL_1[0] = 0
    LOSS_WEIGHT_LOCAL_2 = torch.ones(nclasses_local_2)
    LOSS_WEIGHT_LOCAL_2[0] = 0

    # Class stage mappping. 3 stages to use
    s1_2_s3 = traindataset.l1_2_g
    s2_2_s3 = traindataset.l2_2_g


    local_rank = 0
    num_gpus = torch.cuda.device_count()
    world_size = num_gpus
    print('CUDA available: ', torch.cuda.is_available())
    print('Number of GPUs: ', num_gpus)
    print(f"Running DDP on rank {local_rank}.")
    setup(local_rank, world_size)

    sampler = DistributedSampler(traindataset, num_replicas=world_size, rank=local_rank, shuffle=True, drop_last=False)
    traindataloader = torch.utils.data.DataLoader(traindataset, batch_size=batchsize, 
    num_workers=workers, sampler=sampler, pin_memory=True)


    # Define the model
    if cell == 'lstm':
        network = multistageLSTMSequentialEncoder(24, 24, nstage=stage, nclasses=nclasses,
                                                  nclasses_l1=nclasses_local_1, nclasses_l2=nclasses_local_2,
                                                  input_dim=input_dim, hidden_dim=hidden, n_layers=layer,
                                                  wo_softmax=True)
    else:
        network = multistageSTARSequentialEncoder(24, 24, nstage=stage, nclasses=nclasses,
                                                  nclasses_l1=nclasses_local_1, nclasses_l2=nclasses_local_2,
                                                  input_dim=input_dim, hidden_dim=hidden, n_layers=layer, cell=cell,
                                                  wo_softmax=True)
    network_gt = model_2DConv(nclasses=nclasses, num_classes_l1=nclasses_local_1, num_classes_l2=nclasses_local_2,
                              s1_2_s3=s1_2_s3, s2_2_s3=s2_2_s3,
                              wo_softmax=True, dropout=dropout)

    model_parameters = filter(lambda p: p.requires_grad, network.parameters())
    model_parameters2 = filter(lambda p: p.requires_grad, network_gt.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters]) + sum([np.prod(p.size()) for p in model_parameters2])
    print('Num params: ', params)

    optimizer = torch.optim.Adam(list(network.parameters()) + list(network_gt.parameters()), lr=lr,
                                 weight_decay=weight_decay)
    
    loss = torch.nn.CrossEntropyLoss(weight=LOSS_WEIGHT)
    loss_local_1 = torch.nn.CrossEntropyLoss(weight=LOSS_WEIGHT_LOCAL_1)
    loss_local_2 = torch.nn.CrossEntropyLoss(weight=LOSS_WEIGHT_LOCAL_2)

    if lrS == 1:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1, last_epoch=-1)
    elif lrS == 2:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1, last_epoch=-1)
    elif lrS == 3:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5, last_epoch=-1)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5, last_epoch=-1)


    print('CUDA available: ', torch.cuda.is_available())
    num_gpus = torch.cuda.device_count()
    print('Number of GPUs', num_gpus)
    if torch.cuda.is_available():
        network = network.cuda()
        network_gt = network_gt.cuda()
        
        network = torch.nn.parallel.DataParallel(network, device_ids=list(range(num_gpus)), dim=0)
        network = torch.nn.parallel.DataParallel(network, device_ids=list(range(num_gpus)), dim=0)
        
        loss = loss.cuda()
        loss_local_1 = loss_local_1.cuda()
        loss_local_2 = loss_local_2.cuda()

    start_epoch = 0
    best_test_acc = 0

    if snapshot is not None:
        checkpoint = torch.load(snapshot)
        network.load_state_dict(checkpoint['network_state_dict'])
        network_gt.load_state_dict(checkpoint['network_gt_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizerA_state_dict'])

    if eval_mode:
        print("\n Eval on test set") # NOTE default level is level 3 for evaluate_fieldwise.
        test_acc = evaluate_fieldwise(network, network_gt, testdataset, batchsize=batchsize, prediction_dir=prediction_dir, experiment_id=experiment_id)
        print('Model saved! Best val acc:', test_acc)

    else:
        step_count = stepCount(init_step=0)
        for epoch in range(start_epoch, epochs):
            print("\nEpoch {}".format(epoch+1))
            
            train_epoch(traindataloader, network, network_gt, optimizer, loss, loss_local_1, loss_local_2,
                        lambda_1=lambda_1, lambda_2=lambda_2, lambda_3=lambda_3, lambda_gt=lambda_gt, stage=stage, grad_clip=clip, step_count=step_count,
                        wandb_enable=wandb_enable)

            # call LR scheduler
            lr_scheduler.step()

            #save model
            if checkpoint_dir is not None:
                    checkpoint_name = os.path.join(checkpoint_dir, name + '_epoch_' + str(epoch) + "_model.pth")
                    #best_test_acc = test_acc
                    torch.save({'network_state_dict': network.state_dict(),
                                'network_gt_state_dict': network_gt.state_dict(),
                                'optimizerA_state_dict': optimizer.state_dict()}, checkpoint_name)


            # evaluate model
            if epoch > 1 and epoch % 1 == 0:
                print("\n Eval on test set") # NOTE default level is level 3 for evaluate_fieldwise.
                test_acc = evaluate_fieldwise(network, network_gt, testdataset, batchsize=batchsize, prediction_dir=prediction_dir, experiment_id=experiment_id)
                print('Model saved! Best val acc:', test_acc)

                if wandb_enable:
                    wandb.log({"val_epoch/val_accuracy": test_acc}, step = step_count.step-1)
                        
                if wandb_enable:
                    wandb.summary["best val acc"] = test_acc
                    wandb.summary["best epoch"] = epoch




def train_epoch(dataloader, network, network_gt, optimizer, loss, loss_local_1, loss_local_2, lambda_1,
                lambda_2, lambda_3, lambda_gt, stage, grad_clip, step_count, wandb_enable):

    network.train()
    network_gt.train()

    mean_loss_glob = 0.
    mean_loss_local_1 = 0.
    mean_loss_local_2 = 0.
    mean_loss_gt = 0.
    mean_loss_total = 0.
    
    for iteration, data in enumerate(dataloader):
        optimizer.zero_grad()
        input, target_glob, target_local_1, target_local_2 = data

        if torch.cuda.is_available():
            input = input.cuda()
            target_glob = target_glob.cuda()
            target_local_1 = target_local_1.cuda()
            target_local_2 = target_local_2.cuda()

        output_glob, output_local_1, output_local_2 = network.forward(input)

        # NOTE no mask is passed and no loss is masked. the masking is supposed to be done before the training
        l_glob = loss(output_glob, target_glob)
        l_local_1 = loss_local_1(output_local_1, target_local_1)
        l_local_2 = loss_local_2(output_local_2, target_local_2)

        # TODO verify the lambda here, if it is same as the paper. 
        if stage == 3 or stage == 1:
            total_loss = lambda_3 * l_glob + lambda_1 * l_local_1 + lambda_2 * l_local_2
        elif stage == 2:
            total_loss = l_glob + lambda_2 * l_local_2
        else:
            total_loss = l_glob

        mean_loss_glob += l_glob.data.cpu().numpy()
        mean_loss_local_1 += l_local_1.data.cpu().numpy()
        mean_loss_local_2 += l_local_2.data.cpu().numpy()

        # Refinement -------------------------------------------------
        output_glob_R = network_gt([output_local_1, output_local_2, output_glob])
        l_gt = loss(output_glob_R, target_glob)
        mean_loss_gt += l_gt.data.cpu().numpy()

        # TODO log the loss in wandb during training.
        total_loss = total_loss + lambda_gt * l_gt
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), grad_clip)
        torch.nn.utils.clip_grad_norm_(network_gt.parameters(), grad_clip)
        optimizer.step()

        mean_loss_total += total_loss.data.cpu().numpy()
        metrics_per_step = {"train_step/total_loss": total_loss,
                            "train_step/local_loss_1": l_local_1,
                            "train_step/local_loss_2": l_local_2,
                            "train_step/global_loss": l_glob,
                            "train_step/global_loss_refined": l_gt}
        if wandb_enable:
            wandb.log(metrics_per_step, step=step_count.step)
        if step_count.step%1000 == 0:
            print("step:", step_count.step, "total_loss: %.4f"%(total_loss.data.cpu().numpy()))
        step_count.count()

    mean_loss_local_1 /= (iteration+1)
    mean_loss_local_2 /= (iteration+1)
    mean_loss_glob /= (iteration+1)
    mean_loss_gt /= (iteration+1)
    mean_loss_total /= (iteration+1)

    print('Local Loss 1: %.4f' % (mean_loss_local_1 / iteration))
    print('Local Loss 2: %.4f' % (mean_loss_local_2 / iteration))
    print('Global Loss: %.4f' % (mean_loss_glob / iteration))
    print('Global Loss - Refined: %.4f' % (mean_loss_gt / iteration))

    metrics_per_epoch = {"train_epoch/total_loss": mean_loss_total,
                        "train_epoch/local_loss_1": mean_loss_local_1,
                        "train_epoch/local_loss_2": mean_loss_local_2,
                        "train_epoch/global_loss": mean_loss_glob,
                        "train_epoch/global_loss_refined": mean_loss_gt}
    if wandb_enable:
        wandb.log(metrics_per_epoch, step = step_count.step-1)

if __name__ == "__main__":
    args = parse_args()
    print(args)
    history_step = 0

    model_name = str(args.name) + '_' + str(args.cell) + '_' + str(args.input_dim) + '_' + str(args.batchsize) + '_' + str(
        args.learning_rate) + '_' + str(args.layer) + '_' + str(args.hidden) + '_' + str(args.lrSC) + '_' + str(
        args.lambda_1) + '_' + str(args.lambda_2) + '_' + str(args.weight_decay) + '_' + str(args.fold) + '_' + str(
        args.gt_path) + '_' + str(args.seed)
    print(model_name)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.wandb_enable:
        wandb.init(project='swiss_crop', entity='ozgur', name=f'experiment_{args.experiment_id}',config=args)
    
    main(
        datadir=args.data,
        npz_dir=args.npz_dir,
        data_canton_labels_dir=args.data_canton_labels,
        canton_ids_train=args.canton_ids_train,
        batchsize=args.batchsize,
        workers=args.workers,
        epochs=args.epochs,
        lr=args.learning_rate,
        snapshot=args.snapshot,
        checkpoint_dir=args.checkpoint_dir,
        experiment_id = args.experiment_id,
        prediction_dir = args.prediction_dir,
        weight_decay=args.weight_decay,
        name=model_name,
        layer=args.layer,
        hidden=args.hidden,
        lrS=args.lrSC,
        lambda_1=args.lambda_1,
        lambda_2=args.lambda_2,
        lambda_3=args.lambda_3,
        lambda_gt=args.lambda_gt,
        stage=args.stage,
        clip=args.clip,
        fold_num=args.fold,
        gt_path=args.gt_path,
        cell=args.cell,
        dropout=args.dropout,
        input_dim=args.input_dim,
        apply_cm = args.apply_cm,
        wandb_enable = args.wandb_enable,
        eval_mode = args.eval
    )
    if wandb_enable:
        wandb.finish()
