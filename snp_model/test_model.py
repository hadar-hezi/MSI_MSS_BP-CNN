#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 08:33:09 2020

@author: hadar.hezi@bm.technion.ac.il
"""
import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
import torch.nn.functional as F

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
        for  inputs, labels,sub_labels,paths in prepare.dataloaders['test']:           
            paths = np.array(paths)
            inputs = inputs.to(prepare.device)
            labels = labels.to(prepare.device)
            sub_labels = sub_labels.to(prepare.device)
            outputs = model(inputs)
            prob_y = F.softmax(outputs,1)
            _, preds = torch.max(prob_y, 1)
            # MSI probability is the max value between MSI1 and MSI2
            prob_y_msi = torch.where(prob_y[:,0] > prob_y[:,2],prob_y[:,0],prob_y[:,2])
           
            epoch_preds.append(preds)
            epoch_labels.append(labels)
            epoch_sub_labels.append(sub_labels)
            epoch_paths.append(paths)
            epoch_pos_probs.append(prob_y_msi)
            # calculate accuracy using labels or sub_labels
            num_correct += torch.sum(torch.eq(sub_labels,preds)).item()
        
       
        epoch_preds = torch.cat(epoch_preds)
        epoch_labels= torch.cat(epoch_labels)
        epoch_sub_labels= torch.cat(epoch_sub_labels)
        epoch_pos_probs = torch.cat(epoch_pos_probs)
        epoch_paths = np.hstack(epoch_paths)
        accuracy = (num_correct / len(epoch_preds)) *100
        print("test acc: ", accuracy)
        all_summary = summary(epoch_preds,epoch_labels,epoch_sub_labels,epoch_pos_probs,epoch_paths)
        log_file.save_results(test_data_path, all_summary)
        # calculate AUC
        test_auc = get_auc(all_summary,prepare,hp=hp,mode='test')
    
    return test_auc
