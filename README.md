# Chest X-ray module

This is the code for training the model for ```Module 3``` of this [paper](https://arxiv.org/ftp/arxiv/papers/2003/2003.08605.pdf). 
The paper was admitted to [EMBC 2020](https://embc.embs.org/2020/) and will be published soon.

The pipeline of the system from the paper is below. ```Module 3``` is Abnormality classification. 
![The pipeline](./pics/ProjectIllustration.jpg)

Check [requirements.txt](requirements.txt) for all packages you need.

# training
Download [images](https://nihcc.app.box.com/v/ChestXray-NIHCC) and put all them in the [overall](overall) directory.
To run the training, simply run `python train.py`
You can specify your GPUs in [train.py](train.py)

After in the training ```logs``` directory will be created. The model weights can be found in ```logs/checkpoints```.

# inference

The evaluation Jupyter notebooks of the model on the test and validation data will be published soon.

The AUC score of the model on the test data is provided below.
![aucs](./pics/aucs.png)

# Note
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





