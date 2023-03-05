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
from scipy import stats
from log_file import *
from get_patient_name import *
from hyperparams import *


def plot_patches(cimp_true_cin,title):
    patches_to_plot = 4
    fig, ax = plt.subplots(1,patches_to_plot)
    for i in range(patches_to_plot):
        path = cimp_true_cin.paths[i]
        im = Image.open(f"{path}")
        ax[i].imshow(im)
        snp = cimp_true_cin.rates[i]
        ax[i].title.set_text(f"{snp}")
        ax[i].xaxis.set_visible(False)
        ax[i].yaxis.set_visible(False)
    
    plt.tight_layout()
def plot_cimp_patches(cimp_true_cin,title):
    # fig, ax = plt.subplots(1,4)
    ims=[]
    # fig.suptitle(f"{title} classifications", fontsize=16)
    # print(cimp_true_cin.cimp_types.values())
    i=0
    for c_type in cimp_true_cin.cimp_types.values():
        if c_type.cimp == "GEA CIMP-L":
            continue
        if len(c_type.paths)>0:
            ims.append(c_type.paths[0])
        
    fig, ax = plt.subplots(1,len(ims))    
    for c_type in cimp_true_cin.cimp_types.values():
        if c_type.cimp == "GEA CIMP-L":
            continue
        if len(c_type.paths)==0:
            continue
        path = ims[i]
        im = Image.open(f"{path}")
        ax[i].imshow(im)
        ax[i].title.set_text(f"{c_type.cimp}")
        ax[i].xaxis.set_visible(False)
        ax[i].yaxis.set_visible(False)
        i+=1
    # plt.tight_layout()



def make_boxplot(hp,feature,feature_names,feature_value):
    # read test patches result
    summary = read_results(f"{hp['root_dir']}{hp['test_res_file']}_0.csv")
    preds = summary.preds
    labels = summary.labels
    paths = summary.paths

    # used BCE - labels are flipped. MSI=1 MSS=0         
    # filter true CIN  
    cin_ind = np.array(np.nonzero(labels==0))
    # cin_ind = np.squeeze(cin_ind,1)

    # filter true GS
    gs_ind = np.array(np.nonzero(labels==1))
    # gs_ind = np.squeeze(gs_ind,1)

 
    # get patent name of each patch
    # get SNP value for that patch
    true_cin = snp_res()
    false_gs= snp_res()


    for ind in cin_ind:
        ind = ind.item()
        path = paths[ind]
        name = get_patient_name(path)
        for i,n in enumerate(feature_names):
            if name == n:
                if preds[ind]==labels[ind]:
                    true_cin.paths.append(path)
                    true_cin.rates.append(feature_value[i])

                    # true_cin.append(feature_value[i])
                elif preds[ind]!=labels[ind]:
                    false_gs.paths.append(path)
                    false_gs.rates.append(feature_value[i])
                      # false_gs.append(feature_value[i])
                      # # if len(false_gs_patches)<4:
                      # false_gs_patches.append(path)
                    
               
    true_gs=snp_res()
    false_cin = snp_res()

    for ind in gs_ind:
        ind = ind.item()
        path = paths[ind]
        name = get_patient_name(path)
        for i,n in enumerate(feature_names):
            if name == n:
                if preds[ind]==labels[ind]:
                    true_gs.paths.append(path)
                    true_gs.rates.append(feature_value[i])
                    # true_gs.append(feature_value[i])
                    # if len(true_gs_patches)<4:
                    # true_gs_patches.append(path)
                elif preds[ind]!=labels[ind]:
                    false_cin.paths.append(path)
                    false_cin.rates.append(feature_value[i])
                    # false_cin_feat.append(feature_value[i])
                    # if len(false_cin_patches)<4:
                    # false_cin_patches.append(path)

    # plot_patches(true_cin,"True MSI")  
    # plt.savefig(f"{hp['root_dir']}true_msi.png",dpi=500,bbox_inches='tight')   
    # plot_patches(false_gs,"False MSS")  
    # plt.savefig(f"{hp['root_dir']}false_mss.png", dpi=500,bbox_inches='tight')   
    # plot_patches(true_gs,"True MSS")  
    # plt.savefig(f"{hp['root_dir']}true_mss.png", dpi=500,bbox_inches='tight')   
    # plot_patches(false_cin,"False MSI")  
    # plt.savefig(f"{hp['root_dir']}false_msi.png", dpi=500,bbox_inches='tight')   
         
    data = [true_cin.rates,false_gs.rates,true_gs.rates,false_cin.rates]

 
    print(f"{feature} medians: ", np.median(true_cin.rates),np.median(false_gs.rates),
          np.median(true_gs.rates),np.median(false_cin.rates))
    labels = ["true_mss","false_msi","true_msi","false_mss"]
    
    fig= plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    # rectangular box plot
    ax.grid(True)
    # plt.rcParams.update({'xtick.labelsize':14})
    # plt.rcParams.update({'font.size':14})
    plt.rcParams.update({'xtick.labelsize':16})
    plt.rcParams.update({'ytick.labelsize':16})
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
    # labelsize = 14
    # plt.rcParams.update({'xtick.labelsize':16})
    plt.savefig(f"{hp['root_dir']}boxplot_{feature}.png",dpi=500, bbox_inches = "tight")
    # print(stats.ks_2samp(true_gs_feat,false_cin_feat))
    
def make_cimp_boxplot(hp,feature,feature_names,feature_value):
    # read test patches result
    summary = read_results(f"{hp['test_res_file']}_0.csv")
    preds = np.array(summary.preds)
    labels = np.array(summary.labels)
    paths = summary.paths

    # used BCE - labels are flipped. MSI=1 MSS=0           
    # filter true MSI  
    cin_ind = np.nonzero(labels==1)
    # cin_ind = np.squeeze(cin_ind,1)

    # filter true MSS
    gs_ind = np.nonzero(labels==0)
    # gs_ind = np.squeeze(gs_ind,1)

 
    # cimp values are: CIMP EBV,CIMP-H,CRC CIMP-L,GEA CIMP-L,Non-CIMP
    cimp_true_cin = classified_cimp_type("true_positive")
    cimp_false_gs = classified_cimp_type("false_negative")
    cimp_true_gs = classified_cimp_type("true_negative")
    cimp_false_cin = classified_cimp_type("false_positive")

    
    
    for ind in cin_ind[0]:
        # ind = ind.item()
        path = paths[ind]
        name = get_patient_name(path)
        for i,n in enumerate(feature_names):
            if name == n:
                if preds[ind]==labels[ind]:
                   
                    # if len(true_cin_patches)<4:
                    # true_cin_patches.append(path)
                    cimp_true_cin.cimp_types[feature_value[i]].increase(path)
                    # cimp_true_cin[feature_value[i]]['count'] +=1
                    # cimp_true_cin[feature_value[i]]['path'].append(path.item)
                    # true_cin_feat.append(feature_value[i])
                elif preds[ind]!=labels[ind]:
                    cimp_false_gs.cimp_types[feature_value[i]].increase(path)

                    
    for ind in gs_ind[0]:
        # ind = ind.item()
        path = paths[ind]
        name = get_patient_name(path)
        for i,n in enumerate(feature_names):
            if name == n:
                if preds[ind]==labels[ind]:
                    # true_gs_feat.append(feature_value[i])
                    cimp_true_gs.cimp_types[feature_value[i]].increase(path)
                    # if len(true_gs_patches)<4:
                    # true_gs_patches.append(path)
                elif preds[ind]!=labels[ind]:
                    cimp_false_cin.cimp_types[feature_value[i]].increase(path)


    # plot_cimp_patches(cimp_true_cin,"True MSI")  
    # plt.savefig(f"{hp['root_dir']}true_msi.png",dpi=500,bbox_inches='tight')   
    # plot_cimp_patches(cimp_false_gs,"False MSS")  
    # plt.savefig(f"{hp['root_dir']}false_mss.png", dpi=500,bbox_inches='tight')   
    # plot_cimp_patches(cimp_true_gs,"True MSS")  
    # plt.savefig(f"{hp['root_dir']}true_mss.png", dpi=500,bbox_inches='tight')    
    # plot_cimp_patches(cimp_false_cin,"False MSI")  
    # plt.savefig(f"{hp['root_dir']}false_msi.png", dpi=500,bbox_inches='tight')  
    true_msi_val = list(cimp_true_cin.cimp_types.values())
    false_mss_val = list(cimp_false_gs.cimp_types.values())
    true_msi_counts = [t.count for t in true_msi_val ]
    false_mss_counts =  [t.count for t in false_mss_val ]

    true_msi_counts = np.array(true_msi_counts)
    true_msi_counts = true_msi_counts/np.sum(true_msi_counts)
    false_mss_counts = np.array(false_mss_counts)
    false_mss_counts = false_mss_counts/np.sum(false_mss_counts)
    print(true_msi_counts)
    print(false_mss_counts)
    print("p-value of msi patches: ", stats.ttest_ind(true_msi_counts[1:],false_mss_counts[1:],equal_var=False))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    rcParams.update({'font.size': 14})

    # Multiple bar chart
    labels = ["TN","FP","TP","FN"]
    # cimp_ebv = np.array([cimp_true_gs.cimp_types["CIMP EBV"].count,
    #                      cimp_false_cin.cimp_types["CIMP EBV"].count,
    #                      cimp_true_cin.cimp_types["CIMP EBV"].count,
    #                      cimp_false_gs.cimp_types["CIMP EBV"].count])
    cimp_h = np.array([cimp_true_gs.cimp_types["CIMP-H"].count,
                       cimp_false_cin.cimp_types["CIMP-H"].count,
                       cimp_true_cin.cimp_types["CIMP-H"].count,
                       cimp_false_gs.cimp_types["CIMP-H"].count])
    cimp_crc_l = np.array([cimp_true_gs.cimp_types["CRC CIMP-L"].count,
                           cimp_false_cin.cimp_types["CRC CIMP-L"].count,
                           cimp_true_cin.cimp_types["CRC CIMP-L"].count,
                           cimp_false_gs.cimp_types["CRC CIMP-L"].count])
    cimp_gea_l = np.array([cimp_true_gs.cimp_types["GEA CIMP-L"].count,
                           cimp_false_cin.cimp_types["GEA CIMP-L"].count,
                           cimp_true_cin.cimp_types["GEA CIMP-L"].count,
                           cimp_false_gs.cimp_types["GEA CIMP-L"].count])
    non_cimp = np.array([cimp_true_gs.cimp_types["Non-CIMP"].count,
                         cimp_false_cin.cimp_types["Non-CIMP"].count,
                         cimp_true_cin.cimp_types["Non-CIMP"].count,
                         cimp_false_gs.cimp_types["Non-CIMP"].count])
    
    total = cimp_h +cimp_crc_l +cimp_gea_l +non_cimp
    # print("ebv:",cimp_ebv)
    print("cimp_h:",cimp_h)
    print("cimp_crc_l",cimp_crc_l)
    print("cimp_gea_l",cimp_gea_l)
    print("non_cimp:",non_cimp)
    # cimp_ebv = cimp_ebv / total
    cimp_h = cimp_h / total
    cimp_crc_l = cimp_crc_l / total
    cimp_gea_l = cimp_gea_l / total
    non_cimp = non_cimp / total
    # print("ebv:",cimp_ebv)
    print("cimp_h:",cimp_h)
    print("cimp_crc_l",cimp_crc_l)
    print("cimp_gea_l",cimp_gea_l)
    print("non_cimp:",non_cimp)
    ind = np.arange(4)
    # ax.bar(x=ind, height=cimp_ebv, width=0.55,align='center',color='red')
    ax.bar(x=ind, height=cimp_h, width=0.55,  align='center',color="blue")
    ax.bar(x=ind, height=cimp_crc_l,bottom=cimp_h, width=0.55,  align='center')
    ax.bar(x=ind, height=cimp_gea_l,bottom=cimp_crc_l+cimp_h, width=0.55,  align='center')
    ax.bar(x=ind, height=non_cimp,bottom=cimp_gea_l+cimp_crc_l+cimp_h, width=0.55,  align='center')
    # rcParams['xtick.labelsize'] = 18
    ax.legend(['CIMP-H','CRC CIMP-L','GEA CIMP-L','Non-CIMP'],fontsize=10)

    ax.set_title("Methylation type of classified data",fontsize=18)
    # Define x-ticks
 
    plt.rcParams.update({'xtick.labelsize':14})
    plt.rcParams.update({'ytick.labelsize':14})
  
    # rcParams.update({'font.size': 12})
    plt.xticks(ind,labels)
    # Layout and Display
    
    # plt.tight_layout()
    #plt.show()
    fig.savefig(f"cimp_diagram.png",dpi=500,bbox_inches = "tight")
    true_msi_counts = [t.count for t in true_msi_val ]
    false_mss_counts =  [t.count for t in false_mss_val ]
    fvalue, pvalue = stats.f_oneway(true_msi_counts[1:],false_mss_counts[1:])
    print(fvalue, pvalue)

def snp_and_cimp(hp):
    result=[]
    #load train patients
    df = pd.read_csv(f"{hp['root_dir']}train_patients.csv")
    patients = np.array(df['name'])
    label = np.array(df['label'])
    # SNP data
    path_atlas = f"{hp['root_dir']}dis_comnbined_GIAC_Atlas.csv"
    if os.path.isfile(path_atlas):
        df2 = pd.read_csv(path_atlas)     
        snp_names = list(df2['short_ID'])
        SNP = list(df2['SNP']) 
    # CIMP data
    path_atlas = f"{hp['root_dir']}clinical_Atlas.csv"
    if os.path.isfile(path_atlas):
        df3 = pd.read_csv(path_atlas)     
        atlas_names = list(df3['TCGA Participant Barcode'])
        CIMP = list(df3['Hypermethylation category']) 
        
    for i,patient in enumerate(patients):
        #find SNP value
        for j,name in enumerate(snp_names):
            if name == patient:
                snp_val = SNP[j]
        #find CIMP type 
        for j,name in enumerate(atlas_names):
            if name == patient:
               cimp_type = CIMP[j]
        result.append(patient_values(patient,label[i],snp_val,cimp_type))
    s=[]  
    s_not=[]
    new_labels=[]
    for p in result:
        if p.label == 0:
            if p.cimp == "CIMP-H":
                new_labels.append(2)
                s.append(p.snp)

            else:
                s_not.append(p.snp)
                new_labels.append(1)
    s = np.array(s) 
    s_not=np.array(s_not)      
    print("cimp high msi size: ", len(s))   
    print("cimp not high msi size: ", len(s_not))  
    print("cimp high msi snp < 1200: " ,s[s<1200])
    print("cimp not high msi s<1200: " , s_not[s_not<1200])
    print(new_labels)
    #show pie chart of cimp high and >,< 1200
    y = np.array([len(s[s<1200]),len(s[s>=1200])])
    mylabels = ["<1200", ">1200"]

    plt.pie(y, labels = mylabels)
    plt.title("MSI, CIMP-H patients")
    plt.show() 
    
    #show pie chart of non cimp high and >,< 1200
    y = np.array([len(s_not[s_not<1200]),len(s_not[s_not>=1200])])
    mylabels = ["<1200", ">1200"]

    plt.pie(y, labels = mylabels)
    plt.title("MSI, CIMP low and non CIMP patients")
    plt.show() 
    


    # used BCE - labels are flipped. MSI=1 MSS=0           
    # filter true MSI  
    # msi_ind = np.nonzero(labels==1)

    # filter true MSS
    # mss_ind = np.nonzero(labels==0)
    
    # for each msi patient - check what is the 
    
    
    
def get_cnv_boxplot():
    hp = hyperparams()
    path_atlas = f"{hp['root_dir']}dis_comnbined_GIAC_Atlas.csv"
    if os.path.isfile(path_atlas):
        df = pd.read_csv(path_atlas)     
        atlas_names = list(df['short_ID'])
        feature = list(df['CNV_Fraction_DEL']) 
    make_boxplot(hp,"cnv",atlas_names,feature)
    # make_patient_boxplot(hp,"cnv",atlas_names,feature,preds,labels,names)  
    
def get_purity_boxplot():
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
    
def get_t_cell_boxplot():
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
    path_atlas = f"{hp['root_dir']}dis_comnbined_GIAC_Atlas.csv"
    if os.path.isfile(path_atlas):
        df = pd.read_csv(path_atlas)     
        atlas_names = list(df['short_ID'])
        feature = list(df['SNP']) 
    make_boxplot(hp,"snp",atlas_names,feature)
    # make_patient_boxplot(hp,"snp",atlas_names,feature,preds,labels,names)     
def get_cimp_diagram():
    hp = hyperparams()
    path_atlas = f"clinical_Atlas.csv"
    if os.path.isfile(path_atlas):
        df = pd.read_csv(path_atlas)     
        atlas_names = list(df['TCGA Participant Barcode'])
        feature = list(df['Hypermethylation category']) 
    make_cimp_boxplot(hp,"cimp",atlas_names,feature)
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
    
class cimp_type:
    def __init__(self,cimp_type):
        self.cimp=cimp_type
        self.count=0
        self.paths=[]
    
    def increase(self,path):
        self.count+=1
        self.paths.append(path)
        

class classified_cimp_type:
    def __init__(self,class_res):
        self.class_res = class_res
        self.cimp_types = {"CIMP EBV":cimp_type("EBV"), "CIMP-H":cimp_type("CIMP-H"),
                           "CRC CIMP-L":cimp_type("CRC CIMP-L"),
                           "GEA CIMP-L":cimp_type("GEA CIMP-L"),
                           "Non-CIMP":cimp_type("Non-CIMP")}        

class snp_res:
    def __init__(self):
        self.paths=[]
        self.rates=[]
    
    def updated(self,path,value):
        self.paths.append(path)
        self.rates.append(value)
        
class patient_values():
    def __init__(self,name,label,snp,cimp):
        self.name = name
        self.label = label
        self.snp = snp
        self.cimp = cimp
        
# get_cnv_boxplot()       
# get_purity_boxplot()
# get_age_boxplot()
# get_Leukocyte_boxplot()
# get_t_cell_boxplot()
# get_interferon_boxplot()
# get_cnv_boxplot()
# get_purity_boxplot()
# get_t_cell_boxplot()
#get_snp_boxplot()
hp = hyperparams()
get_cimp_diagram()
# snp_and_cimp(hp)
