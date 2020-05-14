import torch
from torch import nn
from read_data import ChestXrayDataSet
from transforms import transforms_train
from transforms import transforms_val
from torch.utils.data import DataLoader
from catalyst.dl.runner import SupervisedRunner
from model import DenseNet121
from catalyst.contrib.optimizers import RAdam
from torch import optim
from torch.nn import BCEWithLogitsLoss
from catalyst.utils import set_global_seed, prepare_cudnn
from catalyst.dl.callbacks import AUCCallback, AccuracyCallback, F1ScoreCallback
import os
from catalyst.dl import utils

os.environ["CUDA_VISIBLE_DEVICES"]="0,1" #define GPUs for training
SEED = 42
set_global_seed(SEED) 
prepare_cudnn(deterministic=True)
num_classes = 14
PATH_TO_IMAGES ='overall'

BATCH_SIZE = 64
NUM_EPOCHS = 20

train_dataset = ChestXrayDataSet(
    data_dir=PATH_TO_IMAGES,
    image_list_file='labels/train_list.txt',
    transform=transforms_train,
)

val_dataset = ChestXrayDataSet(
    data_dir=PATH_TO_IMAGES,
    image_list_file='labels/test_list.txt',
    transform=transforms_val,
)

loaders = {'train': DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4), 'valid': DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4)}



logdir = "./logs/experiment_name" #where model weights and logs are stored

#define model
model = DenseNet121(num_classes)
model = nn.DataParallel(model)
device = utils.get_device()
print(f"device: {device}")
runner = SupervisedRunner(device=device)


optimizer = RAdam(model.parameters(), lr=1e-4, weight_decay=0.0003)
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
    num_epochs=NUM_EPOCHS,

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
