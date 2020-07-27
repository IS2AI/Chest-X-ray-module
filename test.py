import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from read_data import ChestXrayDataSet
from model import DenseNet121
from catalyst.dl.runner import SupervisedRunner
from torch.utils.data import DataLoader
from catalyst.dl import utils
from transforms import transforms_test
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns







def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default='0', help='choose gpus to train on')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--path_to_images', type=str, default='/workspace/data/chest_paper/overall/images')
    parser.add_argument('--test_list', type=str, default='labels/test_list.txt')
    parser.add_argument('--num_workers',type=int, default=4)
    parser.add_argument('--checkpoint', type=str, default = '/workspace/data/chest_paper/code/logs/20200725_2/checkpoints/best.pth', help='path to checkpoint of the model')
    parser.add_argument('--test_outdir', type=str, default = '/workspace/data/chest_paper/code/outdir', help='directory where metrics of test test will be saved')
    return parser.parse_args()

def sigmoid(x):
    return 1/(1+np.exp(-x))

args = get_args()

os.environ["CUDA_VISIBLE_DEVICES"]=args.gpus

def main():
    N_CLASSES = 14
    CLASS_NAMES = ['Atelectasis', 
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



    # initialize model
    device = utils.get_device()
    model = DenseNet121(N_CLASSES).to(device)
 
    
    
    checkpoint = torch.load(args.checkpoint)

    model.load_state_dict(checkpoint['model_state_dict'])


    # initialize test loader
    test_dataset = ChestXrayDataSet(data_dir=args.path_to_images,
                                    image_list_file=args.test_list,
                                    transform=transforms_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # initialize the ground truth and output tensor
    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()

    # switch to evaluate mode
    
    model.eval()
    with torch.no_grad():
        for i, (inp, target) in enumerate(test_loader):
            target = target.cuda()
            gt = torch.cat((gt, target), 0)
            bs, c, h, w = inp.size()
            input_var = torch.autograd.Variable(inp.view(-1, c, h, w).cuda())
            output = model(input_var)
            output_mean = output.view(bs, -1)
            pred = torch.cat((pred, output_mean.data), 0)

    gt_np = gt.cpu().numpy()
    pred_np = sigmoid(pred.cpu().numpy())

    Y_t = [] #labels for each anomaly
    for i in range(N_CLASSES):
        Y_t.append([])
        for x in gt_np:
            Y_t[i].append(x[i])

    Y_pred = [] #preds for each anomaly
    for j in range(N_CLASSES):
        Y_pred.append([])
        for y in pred_np:
            Y_pred[j].append(y[j])


    AUCs = [] # AUCs for each 
    for i in range(N_CLASSES):
        auc = roc_auc_score(Y_t[i], Y_pred[i])
        AUCs.append(auc)

    matrices=[] #for each
    for i in range(14):
        matrix = confusion_matrix(Y_t[i], np.asarray(Y_pred[i])>0.6)
        matrices.append(matrix)

    
    class_names = ['no disease', 'disease']
    fig = plt.figure(figsize = (20,20))
    for i in range(14):
        plt.subplot(4,4,i+1)
        
        df_cm = pd.DataFrame(
            matrices[i], index=class_names, columns=class_names)
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d").set_title(CLASS_NAMES[i])
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        
        
    plt.show()
    fig.savefig(os.path.join(args.test_outdir,'confusion_matrix1.pdf'))

    fig, axes2d = plt.subplots(nrows=2, ncols=7,
                            sharex=True, sharey=True,figsize = (12, 4))



    for i, row in enumerate(axes2d):
        for j, cell in enumerate(row):
            if i==0:
                x=i+j
            else:
                x=13-i*j
            
            fpr, tpr, threshold = roc_curve(Y_t[x], Y_pred[x])
            roc_auc = auc(fpr, tpr)
                      
            cell.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
            cell.legend(loc = 'lower right', handlelength=0,handletextpad=0,frameon=False, prop={'size': 8})

            cell.plot([0, 1], [0, 1],'r--')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            cell.set_title(CLASS_NAMES[x],fontsize=10)
            
            if i == len(axes2d) - 1:
                cell.set_xlabel('False positive rate')
            if j == 0:
                cell.set_ylabel('True negative rate')
    fig.tight_layout(pad=1.0)    
    plt.show()
    fig.savefig(os.path.join(args.test_outdir,'roc_auc1.pdf'))

if __name__ == '__main__':

    torch.backends.cudnn.benchmark = True
    

    try:
        main()
    except KeyboardInterrupt:
        print('Keyboard Interrupted')