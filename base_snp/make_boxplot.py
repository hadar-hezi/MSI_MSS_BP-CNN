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
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import rcParams
# import seaborn as sns

from log_file import *
from get_patient_name import *
from hyperparams import *



def plot_patches(patches,feature_value,title):
    fig, ax = plt.subplots(1,2)

    fig.suptitle(f"{title} classifications", fontsize=16)
    im = Image.open(f"/tcmldrive/databases/Public/TCGA/{patches[8]}")
    ax[0].imshow(im)
    ax[0].title.set_text(f"SNP {feature_value[8]}")
    plt.axis('off')
    # curr_ax = plt.gca()
    # curr_ax.get_xaxis().set_visible(False)
    # curr_ax.get_yaxis().set_visible(False)
    im = Image.open(f"/tcmldrive/databases/Public/TCGA/{patches[9]}")
    # plt.axis('off')
    ax[1].imshow(im)
    ax[1].title.set_text(f"SNP {feature_value[9]}")
    plt.axis('off')
    # curr_ax = plt.gca()
    # curr_ax.get_xaxis().set_visible(False)
    # curr_ax.get_yaxis().set_visible(False)



def make_boxplot(hp,feature,feature_names,feature_value):
    # read test patches result
    summary = read_results(f"{hp['root_dir']}{hp['test_res_file']}_0.csv")
    preds = summary.preds
    labels = summary.labels
    paths = summary.paths

                
    # filter true CIN  
    cin_ind = np.array(np.nonzero(labels==0))
    # cin_ind = np.squeeze(cin_ind,1)

    # filter true GS
    gs_ind = np.array(np.nonzero(labels==1))
    # gs_ind = np.squeeze(gs_ind,1)

 
    # get patent name of each patch
    # get SNP value for that patch
    true_cin_feat = []
    false_gs_feat = []
    true_cin_patches = []
    false_gs_patches = []


    for ind in cin_ind:
        ind = ind.item()
        path = paths[ind]
        name = get_patient_name(path)
        for i,n in enumerate(feature_names):
            if name == n:
                if preds[ind]==labels[ind]:
                   
                    # if len(true_cin_patches)<4:
                    true_cin_patches.append(path)
                    true_cin_feat.append(feature_value[i])
                elif preds[ind]!=labels[ind]:
                      false_gs_feat.append(feature_value[i])
                      # if len(false_gs_patches)<4:
                      false_gs_patches.append(path)
                    
               
    true_gs_feat = []
    false_cin_feat = []
    true_gs_patches,false_cin_patches = [],[]
    for ind in gs_ind:
        ind = ind.item()
        path = paths[ind]
        name = get_patient_name(path)
        for i,n in enumerate(feature_names):
            if name == n:
                if preds[ind]==labels[ind]:
                    true_gs_feat.append(feature_value[i])
                    # if len(true_gs_patches)<4:
                    true_gs_patches.append(path)
                elif preds[ind]!=labels[ind]:
                    false_cin_feat.append(feature_value[i])
                    # if len(false_cin_patches)<4:
                    false_cin_patches.append(path)

    # plot_patches(true_cin_patches,true_cin_feat,"True MSS")  
    # plt.savefig(f"{hp['root_dir']}true_mss.eps", format='eps')   
    # plot_patches(false_gs_patches,false_gs_feat,"False MSI")  
    # plt.savefig(f"{hp['root_dir']}false_msi.eps", format='eps')   
    # plot_patches(true_gs_patches,true_gs_feat,"True MSI")  
    # plt.savefig(f"{hp['root_dir']}true_msi.eps", format='eps')   
    # plot_patches(false_cin_patches,false_cin_feat,"False MSS")  
    # plt.savefig(f"{hp['root_dir']}false_mss.eps", format='eps')     
         
    data = [true_cin_feat,false_gs_feat,true_gs_feat,false_cin_feat]

 
    print(f"{feature} medians: ", np.median(true_cin_feat),np.median(false_gs_feat), np.median(true_gs_feat),np.median(false_cin_feat))
    labels = ["true_mss","false_msi","true_msi","false_mss"]
    
    fig= plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    # rectangular box plot
    ax.grid(True)
    plt.rcParams.update({'font.size':18})
    bplot = ax.boxplot(data,
                         vert=True, 
                         showmeans=True,# vertical box alignment
                         patch_artist=True,  # fill with color
                         labels=labels)  # will be used to label x-ticks
    ax.set_title(f"{feature} value of classified data",fontsize=24)
    # fill with colors
    colors = ['pink', 'lightblue', 'lightgreen','orange']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    labelsize = 24
    rcParams['xtick.labelsize'] = labelsize
    
    plt.savefig(f"{hp['root_dir']}boxplot_{feature}.png",dpi=300, bbox_inches = "tight")
def make_patient_boxplot(hp,feature,feature_names,feature_value,preds,labels,names):

    feat_values_by_names=[]
    for name in names:
        if name not in feature_names:
            feat_values_by_names.append(np.nan)
        for i,atlas_name in enumerate(feature_names):
            if name==atlas_name:
                feat_values_by_names.append(feature_value[i])
                
                
    # filter true CIN  
    cin_ind = np.array(np.nonzero(labels==0)[0])
    # cin_ind = np.squeeze(cin_ind,1)

    # filter true GS
    gs_ind = np.nonzero(labels==1)[0]
    # gs_ind = np.squeeze(gs_ind,1)
 

 
    # get patent name of each patch
    # get SNP value for that patch
    true_cin_feat = []
    # all cin paths
    # true_cin_paths = paths[cin_ind]
    feat_values_by_names = np.array(feat_values_by_names)

               
    true_gs_feat = []
    
    false_gs_feat = []
    for ind in cin_ind:
        #label is cin
        # if pred is gs
        if not np.isnan(feat_values_by_names[ind]):
            if preds[ind]==labels[ind]:
                true_cin_feat.append(feat_values_by_names[ind])
                
            elif preds[ind]!=labels[ind]:
                false_gs_feat.append(feat_values_by_names[ind])
      
    false_cin_feat = []  
    for ind in gs_ind:
        if not np.isnan(feat_values_by_names[ind]):
            #label is gs
            if preds[ind]==labels[ind]:
                true_gs_feat.append(feat_values_by_names[ind])
            # if pred is cin
            elif preds[ind]!=labels[ind]:
                false_cin_feat.append(feat_values_by_names[ind])      
    
    # true_cin_feat =true_cin_feat[~np.isnan(true_cin_feat)]
    # false_gs_feat =false_gs_feat[~np.isnan(false_gs_feat)]  
    # true_gs_feat =true_gs_feat[~np.isnan(true_gs_feat)]  
    # false_cin_feat =false_cin_feat[~np.isnan(false_cin_feat)]            
    data = [true_cin_feat,false_gs_feat,true_gs_feat,false_cin_feat]
    fig= plt.figure(figsize =(10, 7))
    plt.rcParams.update({'font.size':18})
    ax = fig.add_axes([0, 0, 1, 1])
    ax.grid(True)
    # rectangular box plot
    bplot = ax.boxplot(data,
                         vert=True, 
                         showmeans=True,# vertical box alignment
                         patch_artist=True,  # fill with color
                         labels=labels)  # will be used to label x-ticks
    ax.set_title(f"{feature} value of classified data",fontsize=20)
   
    # # fill with colors
    # colors = ['pink', 'lightblue', 'lightgreen','orange']
    # for patch, color in zip(bplot['boxes'], colors):
    #     patch.set_facecolor(color)
    plt.savefig(f"{hp['root_dir']}patient_boxplot_{feature}.png",dpi=300, bbox_inches = "tight")




def get_cnv_boxplot(preds,labels,names):
    hp = hyperparams()
    path_atlas = f"{hp['root_dir']}dis_comnbined_GIAC_Atlas.csv"
    if os.path.isfile(path_atlas):
        df = pd.read_csv(path_atlas)     
        atlas_names = list(df['short_ID'])
        feature = list(df['CNV_Fraction_DEL']) 
    make_boxplot(hp,"cnv",atlas_names,feature)
    # make_patient_boxplot(hp,"cnv",atlas_names,feature,preds,labels,names)  
    
def get_purity_boxplot(preds,labels,names):
    hp = hyperparams()
    path_atlas = f"{hp['root_dir']}clinical_Atlas.csv"
    if os.path.isfile(path_atlas):
        df = pd.read_csv(path_atlas)     
        atlas_names = list(df['TCGA Participant Barcode'])
        feature = list(df['ABSOLUTE Purity']) 
    make_boxplot(hp,"purity",atlas_names,feature)
    # make_patient_boxplot(hp,"purity",atlas_names,feature,preds,labels,names)       

def get_age_boxplot(preds,labels,names):
    hp = hyperparams()
    path_atlas = f"{hp['root_dir']}clinical_Atlas.csv"
    if os.path.isfile(path_atlas):
        df = pd.read_csv(path_atlas)     
        atlas_names = list(df['TCGA Participant Barcode'])
        feature = list(df['Age at initial pathologic diagnosis']) 

    # make_patient_boxplot(hp,"age",atlas_names,feature,preds,labels,names)  
    
def get_Leukocyte_boxplot(preds,labels,names):
    hp = hyperparams()
    path_atlas = f"{hp['root_dir']}clinical_Atlas.csv"
    if os.path.isfile(path_atlas):
        df = pd.read_csv(path_atlas)     
        atlas_names = list(df['TCGA Participant Barcode'])
        feature = list(df['Leukocyte Fraction']) 

    # make_patient_boxplot(hp,"Leukocyte",atlas_names,feature,preds,labels,names) 
    
def get_t_cell_boxplot(preds,labels,names):
    hp = hyperparams()
    path_atlas = f"{hp['root_dir']}clinical_Atlas.csv"
    if os.path.isfile(path_atlas):
        df = pd.read_csv(path_atlas)     
        atlas_names = list(df['TCGA Participant Barcode'])
        feature = list(df['T cells CD8']) 

    # make_patient_boxplot(hp,"t_cell",atlas_names,feature,preds,labels,names) 
    # 
def get_interferon_boxplot(preds,labels,names):
    hp = hyperparams()
    path_atlas = f"{hp['root_dir']}clinical_Atlas.csv"
    if os.path.isfile(path_atlas):
        df = pd.read_csv(path_atlas)     
        atlas_names = list(df['TCGA Participant Barcode'])
        feature = list(df['Interferon Gamma Response - C1 Hallmark']) 
    make_boxplot(hp,"interferon",atlas_names,feature)
    # make_patient_boxplot(hp,"interferon",atlas_names,feature,preds,labels,names)   

def get_snp_boxplot():
    hp = hyperparams()
    path_atlas = f"{hp['root_dir']}data_table_Colorectal_Adenocarcinoma.csv"
    if os.path.isfile(path_atlas):
        df = pd.read_csv(path_atlas)     
        atlas_names = list(df['patient'])
        feature = list(df['SNP']) 
    make_boxplot(hp,"snp",atlas_names,feature)
    # make_patient_boxplot(hp,"snp",atlas_names,feature,preds,labels,names)     

class summary(NamedTuple):
#     """
    # Saves all loaded labels,paths and calculated probs
    # """

    preds: list
    labels: list
    sub_labels: list
    pos_label_probs: list
    paths: list
    
get_snp_boxplot()       
# get_cnv_boxplot()       
# get_purity_boxplot()
# get_age_boxplot()
# get_Leukocyte_boxplot()
# get_t_cell_boxplot()
# get_interferon_boxplot()