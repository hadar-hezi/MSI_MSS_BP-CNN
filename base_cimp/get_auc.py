#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 12:54:57 2022

@author: hadar.hezi@bm.technion.ac.il
"""


from compute_roc import *



def get_auc(summary,p,hp,mode):   
    labels,probs = roc_per_patient(summary,p,hp,mode)
    fpr, tpr, MSI_auc = compute_roc(labels, probs)
    plot_roc(fpr,tpr,MSI_auc,hp,mode)
    # precision, recall, thresholds = precision_recall_curve(labels, probs)

    return MSI_auc