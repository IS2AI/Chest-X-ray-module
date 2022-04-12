import argparse
import torch
import os
from torch import nn
from read_data import ChestXrayDataSet
from transforms import transforms_train
from transforms import transforms_val
from torch.utils.data import DataLoader
from catalyst.dl.runner import SupervisedRunner
from model import DenseNet121
from catalyst.contrib.nn.optimizers import RAdam
from torch import optim
from torch.nn import BCEWithLogitsLoss
from catalyst.dl.callbacks import AUCCallback, AccuracyCallback, F1ScoreCallback
from catalyst.dl import utils


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default='0', help='choose gpus to train on')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs',type=int, default=20)
    parser.add_argument('--path_to_images', type=str, default='overall')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--log_dir', type=str, default='./logs/experiment_name', help='directory where logs are saved')
    parser.add_argument('--num_workers',type=int, default=4)
    parser.add_argument('--train_list', type=str, default='labels/train_list.txt')
    parser.add_argument('--val_list', type=str, default='labels/val_list.txt')
    return parser.parse_args()


def main():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus 
    SEED = 42
    utils.set_global_seed(SEED) 
    utils.prepare_cudnn(deterministic=True)
    num_classes = 14
    
    #define datasets
    train_dataset = ChestXrayDataSet(
        data_dir=args.path_to_images,
        image_list_file=args.train_list,
        transform=transforms_train,
    )

    val_dataset = ChestXrayDataSet(
        data_dir=args.path_to_images,
        image_list_file=args.val_list,
        transform=transforms_val,
    )

    loaders = {'train': DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers), 'valid': DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=args.num_workers)}


    logdir = args.log_dir #where model weights and logs are stored

    #define model
    model = DenseNet121(num_classes)
    if len(args.gpus)>1:
        model = nn.DataParallel(model)
    device = utils.get_device()
    runner = SupervisedRunner(device=device)


    optimizer = RAdam(model.parameters(), lr=args.lr, weight_decay=0.0003)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.25, patience=2)

    weights = torch.Tensor([10, 100, 30, 8, 
                            40, 40, 330, 140, 
                            35, 155, 110, 250, 
                            155, 200]).to(device)
    criterion = BCEWithLogitsLoss(pos_weight=weights)

    class_names = [
                'Atelectasis',
                'Cardiomegaly',
                'Effusion',
                'Infiltration',
                'Mass',
                'Nodule',
                'Pneumonia',
                'Pneumothorax',
                'Consolidation',
                'Edema',
                'Emphysema',
                'Fibrosis',
                'Pleural_Thickening',
                'Hernia']

    runner.train(
        model=model,
        logdir=logdir,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        num_epochs=args.epochs,

        # We can specify the callbacks list for the experiment;
        # For this task, we will check AUC and accuracy
        callbacks=[

            AUCCallback(
                input_key="targets",
                output_key='logits',
                prefix='auc',
                class_names=class_names,
                num_classes=num_classes,
                activation='Sigmoid',
            ),
            
            AccuracyCallback(
                input_key = "targets",
                output_key = "logits",
                prefix = "accuracy",
                accuracy_args = [1],
                num_classes = 14,
                threshold = 0.5,
                activation = 'Sigmoid',
            ),
        ],
        main_metric='auc/_mean',
        minimize_metric=False,
        verbose=True,
    )


if __name__ == '__main__':

    torch.backends.cudnn.benchmark = True
    
    try:
        main()
    except KeyboardInterrupt:
        print('Keyboard Interrupted')
