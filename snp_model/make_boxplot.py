#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 13:57:33 2021

@author: hadar.hezi@bm.technion.ac.il
"""
from typing import NamedTuple
import torch
import os 
import pandas as pd

from log_file import *
from get_patient_name import *
from hyperparams import *


hp = hyperparams()
# read test patches result
summary = read_results(hp['test_res_file'])
preds = summary.preds
labels = summary.labels
paths = summary.paths
path_snp = f"{hp['root_dir']}SNP_per_patient_COAD_READ_Atlas.csv"
if os.path.isfile(path_snp):
    df = pd.read_csv(path_snp)     
    snp_names = list(df['patient'])
    snp = list(df['SNP'])
# filter true MSI
msi_ind = torch.nonzero(labels==1)
true_msi = torch.nonzero(preds[msi_ind] == labels[msi_ind])
# filter true MSS
mss_ind = torch.nonzero(labels==0)
true_mss = torch.nonzero(preds[mss_ind] == labels[mss_ind])
# filter false MSS
false_mss = torch.nonzero(preds[msi_ind] != labels[msi_ind])
# get patent name of each patch
# get SNP value for that patch
true_msi_snp = []
for ind in true_msi:
    path = paths[ind]
    name = get_patient_name(path)
    for i,snp_name in enumerate(snp_names):
        if name == snp_name:
           true_msi_snp.append(snp[i])
           
true_mss_snp = []
for ind in true_mss:
    path = paths[ind]
    name = get_patient_name(path)
    for i,snp_name in enumerate(snp_names):
        if name == snp_name:
           true_mss_snp.append(snp[i])
           
false_mss_snp = []
for ind in false_mss:
    path = paths[ind]
    name = get_patient_name(path)
    for i,snp_name in enumerate(snp_names):
        if name == snp_name:
           false_mss_snp.append(snp[i])
           
data_dict = {'true_msi':true_msi_snp,'true_mss':true_mss_snp,'false_mss':false_mss_snp}
df = pd.DataFrame(data_dict)   
boxplot = df.boxplot(column=['true_msi', 'true_mss', 'false_mss'])

# plot boxplot

    
class summary(NamedTuple):
#     """
    # Saves all loaded labels,paths and calculated probs
    # """

    preds: list
    labels: list
    sub_labels: list
    pos_label_probs: list
    paths: list
        