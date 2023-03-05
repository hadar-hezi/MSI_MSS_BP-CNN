#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 09:08:06 2021

@author: hadar.hezi@bm.technion.ac.il
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, Sampler, TensorDataset
import matplotlib.pyplot as plt
from collections import namedtuple
import sklearn.metrics as sk
import pandas as pd
import time
import os
import copy
import sys
import statistics          
import random


from prepare_data import Prepare
from test_model import test_model
from train_model import train_model
from hyperparams import hyperparams

# boolean whether to run fold
def get_fold(k,curr_i):
    if curr_i not in k:
        return False
    else: return True
    
# which transfer learning model to use
def set_model(hp):
    if hp['model_type'] == 'resnext':
        model_ft = torch.hub.load('pytorch/vision:v0.6.0', 'resnext50_32x4d', pretrained=hp['pretrained'])
    elif hp['model_type'] == 'inception':
        model_ft = torch.hub.load('pytorch/vision:v0.6.0', 'inception_v3', pretrained=hp['pretrained'])
    elif hp['model_type'] == 'efficient':
        model_ft = EfficientNet.from_pretrained('efficientnet-b7',num_classes=hp['num_classes'])
        
    if hp['unfixed'] == False:
        model_ft = p.train_params(hp,model_ft)   
    # define classifier layers
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(
        nn.Linear(num_ftrs, 100),
        nn.Linear(100, hp['num_classes'])
        )
    model_ft = model_ft.to(p.device)
    if  hp['optimizer_type'] == 'SGD':
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=hp['lr'],momentum=0.99)
    elif  hp['optimizer_type'] == 'Adam':
        optimizer_ft = optim.Adam(model_ft.parameters(), lr=hp['lr'])
    return model_ft, optimizer_ft

def seed_torch(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_deterministic(True)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

torch.cuda.empty_cache()
# hyper-parameters dictionary
hp = hyperparams()
# set seeds for reproducity
seed_torch( hp['newseed'] )
# define criterion function
criterion = nn.BCELoss()
# cross validation iterations
cross_valid =  hp['n_folds']                         
print(hp)
# object for data and model definitions
p = Prepare(hp)
for i in range(cross_valid):
    saved_state = None
    hp['curr_fold'] = i    
    train_bool = get_fold(hp['folds'],i)      
    # will get the next fold
    p.prepare_data(hp)     
    print(p.device)
    # create the data loaders
    p.create_train_validation_loaders()                                
    if(train_bool):
        model_ft,optimizer_ft = set_model(hp)
        # training loop
        model_ft,best_valid_auc = train_model(model_ft, criterion, optimizer_ft,p,hp,saved_state,early_stopping=5)
        torch.cuda.empty_cache()
        #create the test loadrer
        p.create_test_loader()
        # the test loop
        test_auc = test_model(model_ft,p,hp)
        # load best model AUC
        p.create_test_loader()
        model_ft,optimizer_ft = set_model(hp)
        # load previously trained model
        if os.path.isfile(f"{hp['root_dir']}{hp['checkpoint_save']}_{i}.pt"):   
            model_ft,_  = p.load_model(f"{hp['root_dir']}{hp['checkpoint_save']}_{i}.pt",model_ft)
        test_auc = test_model(model_ft,p,hp)

    
