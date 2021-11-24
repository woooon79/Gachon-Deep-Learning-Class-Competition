# Gachon-Deep-Learning-Class-Competition
Professor Jungchan Cho's Deep Learning Class (Spring, 2021) classification competition.  
We used a subset of Yoga-82 dataset: https://sites.google.com/view/yoga-82/  
The testset consists of 600 images and 6 classes: [balancing, inverted, reclining, sitting, standing, and wheel.]  

# Tutorial


To run this tutorial, please make sure the install and import following packages.
```
import torch
import os
import cv2
import csv
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import PIL.ImageOps
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
import random
import pandas as pd
```

## Initial setting
Setting to use GPU and set various path to load datasets.

```
#Set to use gpu
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
EPOCH = 10
BATCH_SIZE = 32

#Set test and train images path
base_dir = '../input/'
train_dir = os.path.join(base_dir, 'train-dataset')
test_dir = os.path.join(base_dir,'test-dataset')
```
## Custom Dataset
Pytorch provides useful tools such as torch.utils.data.Dataset and torch.utils.data.DataLoader to make it easier to handle datasets.  This simplifies mini batch learning, data shuffle, and parallelism. The default usage is to define Dataset and forward it to the DataLoader. But you can create your own custom datasets by inheriting torch.utils.data.Dataset. torch.utils.data.Dataset is an abstract class that provides datasets in Pytorch. Inherit Dataset and override the following methods to create a custom dataset:


## Training and Test Function
Define functions to train and test. Train function’s parameters are model, dataloader, optimizer, epoch. The epoch and optimizer values are setting by Mytuner class. Test function’s parameters are model which is trained by train function and dataloader. 
  

## Load datasets and setting the model.
We used the yoga posture dataset and the resnet18 model. You can get user’s input to choose pretrained or non-pretrained. If user wants non-pretrained model, they will input ‘False’.  
  

## Improve the performance
We attempt several methods to improve the performance of the model. A variety of other methods can be applied.
  
  
### + Data Augmetation
```
# Data augmentation --------------------------------------------
transform = transforms.Compose([transforms.ToTensor(),  
                                transforms.RandomHorizontalFlip() ,
                                transforms.Normalize((0.1307,), (0.3081,))])
````

Due to the characteristics of yoga posture, Random Vertical Flip was not used for data augmentation, but only Random Horizontal Flip was used.

### + Modify fully-connected layer
```
# Transform the fully connected layer
model.fc = nn.Sequential(nn.Linear(num_ftrs, 512), # attach trainable classifier
                                      nn.ReLU(),
                                      nn.Dropout(0.3),
                                      nn.Linear(512, 6)) 

```
Instead of immediately reducing the channels to 1024 to 6, we added a ReLU activation function and Dropout to increase the performance.  

## Hyperparameter tuning (Mytuner class)  
> Mytuner(model,train_set,val_set,config)    

  Mytuner will randomly select values from a list of hyperparameter values in the config dictionary to create a best combination. There are five hyperparameters we used for tuning: Epoch, Batch size, learning rate, momentum, and optimizer.    For each parameter, various candidate values are stored in the ‘config’ dictionary in the form of a list. Only ‘lr’ key values are range value([min,max]). You can organize the list with the values you want. 

``` # Make the elements to be tuned into a dictionary form.
config = { 
         'n_epochs' : [3], 
         'batch' : [4,8,16,32,64,128],
         'lr' : [1e-5,1e-4], 
         'momentum' : [0.4,0.6,0.8,0.9,0.95,0.99],
         'optimizer': [optim.RMSprop, optim.Adam]
         }
```

  
You can declare Mytuner with 4 parameters (model, train dataset, validation dataset, config).  
  
>  Parameters
> *	model : Target model to use for learning.
> *	train_set  : Train dataset from which to load the data. It will be Custom Dataset, created by inheriting torch.utils.data.dataset and overriding methods. torch.utils.data.Dataset is an abstract class that provides datasets in Pytorch.
> *	val_set : Test dataset from which to load the data. It will be Custom Dataset, created by inheriting torch.utils.data.dataset and overriding methods. torch.utils.data.Dataset is an abstract class that provides datasets in Pytorch.
> *	config : 
>     +	n_epochs([int,…]) : Number of epochs to try for each parameter combination
>     +	batch([int,…]) : List the batch sizes you want to try.
>     +	lr([float,float]) : Input learning rates’ range value in list. ex) [min,max]
>     +	momentum([float,…]) : List the various momentum value you want to try. (if the model has momentum parameters)
>     +	optimizer([optim.XXX,…]) : List the optimizers you want to try.

Then call start function in Mytuner with the number of hyperparameters combination trial. 
Tuning, training and validation are all going in this process . We set the number of hyperparameters combination to 20, but you can use a different number.  

## Test 
Load the test dataset and proceed with the test. Classification results are saved in the csv file format.

       
         
    
# Open Source SW Contribution
> Subject: Deep Learning  
> Subject ID: 13177001  
> Professor Name: Jungchan Cho  
> SW Developer(s): 김수현 이정준 윤호운 전수환  
> Date: 04/03/2021 ~ 06/04/2021  

**URL (Kaggle)**

https://www.kaggle.com/kimsuhyeon01/team-h-problem1  


**Target Library**  

Pytorch  


**Title (class/function name)**

Mytuner(model,train_set,val_set,config)

**Description**

Hyperparameter tuning, such as choosing different learning rates or optimization methods, can often have a significant impact on model performance. We implement an automatic tuner ‘Mytuner’ class to help to find the best combination of parameters. We selected random search method to tune. Random Search can find the optimal hyperparameter value faster while drastically reducing the number of unnecessary iterations. You can declare Mytuner with 4 parameters (model, train dataset, validation dataset, config). Config is a dictionary form which contains hyperparameters to tune (number of epoch, batch size, learning rate, momentum, optimizer). Then call start function in Mytuner with the number of hyperparameters combination trial. Tuning, training and validation are all going in this process 

