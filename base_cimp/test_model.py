#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 08:33:09 2020

@author: hadar.hezi@bm.technion.ac.il
"""
import torch
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, Sampler, TensorDataset
import matplotlib.pyplot as plt
from collections import namedtuple
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support                                                           
import torch.nn.functional as F
import os
import copy
import sys

import log_file
from log_file import summary
from get_auc import *

def test_model(model,prepare,hp): 
    model.eval()
    num_correct = 0
    param_dict = {}
    epoch_preds = []
    epoch_paths = []
    epoch_labels = [] 
    epoch_sub_labels=[]
    epoch_pos_probs=[]
    test_data_path =  f"{hp['root_dir']}/{hp['test_res_file']}_{hp['curr_fold']}.csv"
      
    with torch.no_grad():
        for  inputs, labels,paths in prepare.dataloaders['test']:           
            paths = np.array(paths)
            inputs = inputs.to(prepare.device)
            labels = labels.to(prepare.device)
            # transfer to MSI =1 MSS=0 for binary loss
            labels = 1-labels
            outputs = model(inputs)
            prob_y = F.softmax(outputs,1)
            _, preds = torch.max(prob_y, 1)
            preds = 1-preds
            #MSI probability
            prob_y = prob_y[:,0]
           
            epoch_preds.append(preds)
            epoch_labels.append(labels)
            epoch_paths.append(paths)
            epoch_pos_probs.append(prob_y)
            # calculate accuracy using labels or sub_labels
            num_correct += torch.sum(torch.eq(labels,preds)).item()
        accuracy = (num_correct / prepare.dataset_sizes['test']) *100
        print("test acc: ", accuracy)
        epoch_preds = torch.cat(epoch_preds)
        epoch_labels= torch.cat(epoch_labels)
        epoch_pos_probs = torch.cat(epoch_pos_probs)
        epoch_paths = np.hstack(epoch_paths)
        all_summary = summary(epoch_preds,epoch_labels,epoch_pos_probs,epoch_paths)
        log_file.save_results(test_data_path, all_summary)
        # area under the curve patient-level
        valid_auc = get_auc(valid_summary,prepare,mode='valid',hp=hp)
    return auc_


