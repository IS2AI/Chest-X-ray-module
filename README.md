# Chest X-ray module

This is the code for training the model for ```Module 3``` of this [paper](https://arxiv.org/ftp/arxiv/papers/2003/2003.08605.pdf). 
The paper was admitted to [EMBC 2020](https://embc.embs.org/2020/) and will be published soon.

The pipeline of the system from the paper is below. ```Module 3``` is Abnormality classification. The visualization was done using [Grad-Cam](https://arxiv.org/abs/1610.02391).
![The pipeline](./pics/ProjectIllustration.jpg)


## Install
```pip install -r requirements.txt```

## Train
Download [images](https://nihcc.app.box.com/v/ChestXray-NIHCC) and put all of them in the [overall](overall) directory.
To run the training with default options, simply run `python train.py`
#### Flags
- `--gpus`: IDs of GPUs to train on.
- `--batch_size`: Number of samples that will be propagated through the network in one forward/backward pass.
- `--epochs`: Number of epochs to train the model.
- `--path_to_images`: Directory where images are stored.
- `--lr`: Learning rate.
- `--log_dir`: Directory where logs and wieghts will be saved.
- `--num_worker`: Positive integer will turn on multi-process data loading with the specified number of loader worker processes (Check PyTorch [docs](https://pytorch.org/docs/stable/data.html)).
- `--train_list`: Path to a file with image names and labels of the train data. 
- `--val_list`: Path to a file with image names and labels of the validation data. 

After in the training ```logs``` directory will be created. The model weights can be found in ```logs/experiment_name/checkpoints```.

## Test
Download our model weights file to run inference ([Google Drive](https://drive.google.com/drive/folders/1sW36FwQgA2Qan5O1DVRzjh0hZ5cefG_U?usp=sharing)) and place it into [checkpoints](checkpoints).
To run the inference on the test data with default options and our weights, simply run `python test.py`
You can check your own model by setting a path to your `.pth` file by `-- checkpoint`.
The visualisation of the metrics in the pdf format will be saved in a directory defined by `--test_outdir`.
#### Flags
- `--gpus`: IDs of GPUs to run inference on.
- `--batch_size`: Batch size of test loader.
- `--path_to_images`: Directory where images are stored.
- `--test_list`: Path to a file with image names and labels of the test data.
- `--num_worker`: Positive integer will turn on multi-process data loading with the specified number of loader worker processes (Check PyTorch [docs](https://pytorch.org/docs/stable/data.html)).
- `--checkpoint`: Path to a checkpoint of the model (weights).
- `--test_outdir`: Directory where visualisation of the metrics in the pdf format will be saved.

The AUC score of the model on the test data is provided below.

![aucs](./pics/aucs.png)


*The training algorithm is not deterministic, which means that the results may be slightly diffrenet even the model was trained on the same data and with the same hyperparameters.*
## Note
If you use this code in research, please cite the following paper:
```
@misc{2003.08605,
Author = {Kudaibergen Urinbayev and Yerassyl Orazbek and Yernur Nurambek and Almas Mirzakhmetov and 
          Huseyin Atakan Varol},
Title = {End-to-End Deep Diagnosis of X-ray Images},
Year = {2020},
Eprint = {arXiv:2003.08605},
}
```


