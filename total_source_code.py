"""
Created on Fri Jun  4 10:42:57 2021
##### Hyperparameter Auto-tuner #####
@author: Yoon howoon, Lee JeongJun, Kim suhyeon, Jeon suhwan
"""
#Set to use gpu
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
EPOCH = 10
BATCH_SIZE = 32

#Set test and train images path
base_dir = '../input/'
train_dir = os.path.join(base_dir, 'train-dataset')
test_dir = os.path.join(base_dir,'test-dataset')

# Import libraries ----------------------------------------
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

# Data augmentation --------------------------------------------
transform = transforms.Compose([transforms.ToTensor(),  
                                transforms.RandomHorizontalFlip() ,
                                transforms.Normalize((0.1307,), (0.3081,))])

# Custom Dataset ----------------------------------------
class MyDataset(Dataset):
  def __init__(self, image_dir, label, transform=None, test = False):
    self.image_dir = image_dir
    self.label = label
    self.image_list = os.listdir(self.image_dir)
    self.transform = transform
    self.test_mode = test
  def __len__(self):
    return len(self.image_list)

  def __getitem__(self,idx):
    image_name = os.path.join(self.image_dir, self.image_list[idx])
    image = cv2.imread(image_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224,224), cv2.INTER_AREA)
    image = transform(image)

    if self.test_mode:
      return(image, self.image_list[idx])
    else:
      return(image, self.label)


# Training ----------------------------------------------------
def train(model, train_loader, optimizer, epoch):
  model.train()
  for i, (image, target) in enumerate(train_loader):
    image, target = image.cuda(), target.cuda()

    output = model(image)

    optimizer.zero_grad()
    loss = F.cross_entropy(output, target).cuda()
    
    loss.backward()
    optimizer.step()
    
    if i % 1000 ==0: # 1000 iteration 마다 출력
      print('Train Epoch : {} [{}/{} ({:.0f})%]\tLoss: {:.6f}'.format(epoch, i*len(image),len(train_loader.dataset), 100.*i / len(train_loader),loss.item()))

# Validation ----------------------------------------------------
def evaluate(model, val_loader):
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for (image, target) in val_loader:
      image, target = image.cuda(), target.cuda()
      # image = model_sg(image)

      output = model(image)

      test_loss += F.cross_entropy(output, target, reduction='sum').item()
      pred = output.max(1, keepdim=True)[1]
      correct += pred.eq(target.view_as(pred)).sum().item()

  test_loss /= len(val_loader.dataset)
  test_accuracy = 100. * correct / len(val_loader.dataset)
return test_loss, test_accuracy

#load dataset
balancing_train = MyDataset(os.path.join(train_dir, "balancing_train"),0,transforms)
inverted_train = MyDataset(os.path.join(train_dir, "inverted_train"),1,transforms)
reclining_train = MyDataset(os.path.join(train_dir,"reclining_train"),2,transforms)
sitting_train = MyDataset(os.path.join(train_dir,"sitting_train"),3,transforms)
standing_train = MyDataset(os.path.join(train_dir,"standing_train"),4,transforms)
wheel_train = MyDataset(os.path.join(train_dir,"wheel_train"),5,transforms)

balancing_valid = MyDataset(os.path.join(train_dir,"balancing_valid"),0,transforms)
inverted_valid = MyDataset(os.path.join(train_dir,"inverted_valid"),1,transforms)
reclining_valid = MyDataset(os.path.join(train_dir,"reclining_valid"),2,transforms)
sitting_valid = MyDataset(os.path.join(train_dir,"sitting_valid"),3,transforms)
standing_valid = MyDataset(os.path.join(train_dir,"standing_valid"),4,transforms)
wheel_valid = MyDataset(os.path.join(train_dir,"wheel_valid"),5,transforms)


# Declare train_set, val_set, model, config ----------------------------------------------------
train_set = ConcatDataset([balancing_train, inverted_train, reclining_train, sitting_train, standing_train, wheel_train])
val_set = ConcatDataset([balancing_valid, inverted_valid, reclining_valid, sitting_valid, standing_valid, wheel_valid])
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features

# Transform the fully connected layer
model.fc = nn.Sequential(nn.Linear(num_ftrs, 512), # attach trainable classifier
                                      nn.ReLU(),
                                      nn.Dropout(0.3),
                                      nn.Linear(512, 6)) 
model = model.cuda()

# Make the elements to be tuned into a dictionary form.
config = {'seed' : 0, 
         'n_epochs' : [3], 
         'batch' : [4,8,16,32,64,128],
         'lr' : [1e-7,1e-4], 
         'momentum' : [0.4,0.6,0.8,0.9,0.95,0.99],
         'optimizer': [optim.RMSprop, optim.Adam]
         }


# Mytuner class ----------------------------------------------------
class Mytuner:
    def __init__(self,model,train_set,val_set,cfg): # input parameters
        self.model = model
        self.train_set = train_set
        self.val_set = val_set
        self.cfg = cfg
       
    def start(self, n_trys): # 
      total = []
      total_best = 0
      for n_try in range(n_trys):
        best = 0
        model = copy.deepcopy(self.model)

        # randomize the parameters ==========================================================================================================
        optimizer_n = random.randrange(len(self.cfg['optimizer']))
        epoch_n = random.randrange(len(self.cfg['n_epochs']))
        lr_value = random.uniform(self.cfg['lr'][0], self.cfg['lr'][-1])
        momentum_n = random.randrange(len(self.cfg['momentum']))
        batch_n = random.randrange(len(self.cfg['batch']))

        train_loader = DataLoader(self.train_set, batch_size=self.cfg['batch'][batch_n], shuffle=True)
        val_loader = DataLoader(self.val_set, batch_size=self.cfg['batch'][batch_n], shuffle=False)

        if self.cfg['optimizer'][optimizer_n] == optim.Adam:
          optimizer = self.cfg['optimizer'][optimizer_n](model.parameters(), lr= lr_value)
        else:
          optimizer = self.cfg['optimizer'][optimizer_n](model.parameters(), lr= lr_value, momentum=self.cfg['momentum'][momentum_n]) 
        # =====================================================================================================================================

        
        for epoch in range(1, self.cfg['n_epochs'][epoch_n] + 1):
          train(model, train_loader, optimizer, epoch)
          test_loss, test_accuracy = evaluate(model, val_loader)

          if test_accuracy > total_best: # Save the best model
            torch.save(model.state_dict(), "./best_model.pth")
            total_best = test_accuracy
            print('best_model save')
          
          if test_accuracy > best: # Create a dataframe with the best model in one epoch
            best = test_accuracy
            best_dict = {
                'epoch' : self.cfg['n_epochs'][epoch_n],
                'batch' : self.cfg['batch'][batch_n],
                'momentum' : self.cfg['momentum'][momentum_n],
                'optimizer' : self.cfg['optimizer'][optimizer_n],
                'lr' : lr_value,
                'test_loss' : test_loss,
                'test_accuracy' : test_accuracy
            }
          print('[{}] Test Loss : {:.4f}, Accuracy : {:.4f}%'.format(epoch, test_loss, test_accuracy))
        total.append(best_dict)
        print('finish train')
        print('best acc: ', best)
        df = pd.DataFrame(total)
        df.to_csv("/content/param.csv", mode='w') # Store trained parameter combinations and results in csv
        print(df)


# Test---------------------------------------------------------------------
import csv

load=torch.load('./best_model.pth')
model.load_state_dict(load)
model.eval()

print("load model for test set")

f=open('./prediction.csv','w',newline='')
w=csv.writer(f)
w.writerow(['id','target'])

test_set=MyDataset(test_dir,0,transforms,test=True)
test_loader=DataLoader(test_set,batch_size=BATCH_SIZE,shuffle=False)

preds=[]
img_ids=[]
correct=0

with torch.no_grad():
    for (image,image_name) in test_loader:
        image=image.to(DEVICE)
        output=model(image)
        
        pred=output.max(1,keepdim=True)[1]
        preds.extend(pred)
        img_ids.extend(image_name)
for i in range(600):
    w.writerow([img_ids[i][:-4],str(preds[i].item())])
    
f.close()
print("save prediction csv")

