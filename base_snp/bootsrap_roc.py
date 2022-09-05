#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 23:48:31 2021

@author: hadar.hezi@bm.technion.ac.il
"""

from __future__ import print_function, division

import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from sklearn.metrics import roc_curve, auc, roc_auc_score
import os
import sys

# PACKAGE_PARENT = '/tcmldrive/hadar/MSIMSS_pytorch'
# SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
# sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from prepare_data import Prepare
from log_file import log_file
# from visualize_model 
from compute_roc import *
# from matlab_data import *


def bootstrap_roc(N_boot, summary,p,mode,hp):
    # rng = np.random.RandomState(42)
    total_tpr =[]
    total_fpr=[]
    total_auc=[]
    # TODO:
    # do a shuffle on the indices and take 95% of then at each iteration
    for j in range(N_boot):
        # indices = rng.randint(0, len(preds), len(preds))
      # boot_preds = preds[indices]
      #   boot_labels = labels[indices]
      #   boot_paths = paths[indices]  
        labels,probs = roc_per_patient(summary,p,hp,mode)
        #prec, recall, thresholds = precision_recall_curve(labels,probs)
        #plot_prec(recall,prec, f1,hp,mode)
        lr_fpr, lr_tpr, MSI_tp_auc = compute_roc(labels, probs)
                                                
        print("iter ", j, "AUC: ", MSI_tp_auc)
    
        mean_fpr = np.linspace(0, 1, 200)
        # mean_fpr = lr_fpr
        interp_tpr = np.interp(mean_fpr, lr_fpr, lr_tpr)
        interp_tpr[0] = 0.0
        total_tpr.append(interp_tpr)
        total_fpr.append(lr_fpr)
        total_auc.append(MSI_tp_auc)
    
    mean_tpr = np.mean(total_tpr, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(total_auc)
    std_tpr = np.std(total_tpr, axis=0)
    # tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    # tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    print(mean_auc)
    sorted_auc = np.array(total_auc)
    sorted_auc.sort()
    
    # Computing the lower and upper bound of the 90% confidence interval
    # You can change the bounds percentiles to 0.025 and 0.975 to get
    # a 95% confidence interval instead.
    confidence_lower = sorted_auc[int(0.05 * len(sorted_auc))]
    # nonzero returns tuple
    index = np.nonzero(total_auc==confidence_lower)
    index_lo = index[0]
    
    # take the first occurenrce from indices list
    tpr_lo_ci = total_tpr[index_lo[0]]
    fpr_lo_ci = total_fpr[index_lo[0]]
    
    confidence_upper = sorted_auc[int(0.95 * len(sorted_auc))]
    index= np.nonzero(total_auc==confidence_upper)
    index_hi = index[0]
    tpr_hi_ci = total_tpr[index_hi[0]]
    fpr_hi_ci = total_fpr[index_hi[0]]
    
    fpr = [mean_fpr,mean_fpr,mean_fpr]
    tpr = [tpr_lo_ci,mean_tpr,tpr_hi_ci]
    auc_ = [confidence_lower,mean_auc,confidence_upper]
    plot_roc_with_ci(fpr,tpr,auc_,hp,mode)
    return mean_auc