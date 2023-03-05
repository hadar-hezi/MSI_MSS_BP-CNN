from log_file import *
from hyperparams import hyperparams
# from visualize_model import visualize_model
from prepare_data import *
from compute_roc import *
import pickle
import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
from scipy import io
import pandas as pd
# from matlab_data import *


    
    
def plot_mean_auc(p,mode,hp):
    total_tpr =[]
    total_fpr=[]
    total_auc=[]
    for j in range(hp['n_folds']):
        try:
            with (open(f"{hp['root_dir']}roc_out_{j}.p", "rb")) as openfile:
            # with (open(f"{root}roc_out_pca_{j}.p", "rb")) as openfile:
                data = pickle.load(openfile)
                labels = np.array(data['labels'])
                probs = np.array(data['probs'])
        except:
            print(f"{hp['root_dir']}/{hp['test_res_file']}_{j}.csv")
            summary = read_results(f"{hp['root_dir']}/{hp['test_res_file']}_{j}.csv")
            labels,probs = roc_per_patient(summary,p,hp,mode)
            
            #pickle data
            data = {"labels": labels, "probs": probs}
            pickle.dump( data, open( f"{hp['root_dir']}roc_out_{j}.p", "wb" ) )
        
        lr_fpr, lr_tpr, MSI_tp_auc = compute_roc(labels, probs)
                                                
        print("iter ", j, "AUC: ", MSI_tp_auc)
    
        mean_fpr = np.linspace(0, 1, 200)
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
    print(auc_)
    plot_roc_with_ci(fpr,tpr,auc_,hp,mode)
    return mean_auc

    
def get_mean_roc(root):    
    total_tpr =[]
    total_fpr=[]
    total_auc=[]
    for j in range(5):
        with (open(f"{root}roc_out_{j}.p", "rb")) as openfile:
            data = pickle.load(openfile)
        labels = data['labels']
        probs = data['probs']
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
    return fpr,tpr,mean_auc,total_auc
    
def roc_boxplot(aucs_base,aucs_sub,feature):
    data = [aucs_base,aucs_sub]

    print("medians: ", np.median(aucs_base),np.median(aucs_sub))
    labels = ["baseline",f"{feature}"]
    labelsize = 22
    mpl.rcParams['xtick.labelsize'] = labelsize
    mpl.rcParams['font.size'] = 14
    mpl.rcParams['axes.titlesize'] = labelsize
    fig= plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    # rectangular box plot
    bplot = ax.boxplot(data,
                         vert=True, 
                         showmeans=True,# vertical box alignment
                         patch_artist=True,  # fill with color
                         labels=labels)  # will be used to label x-ticks
    ax.set_ylim([0.67, 0.9])
    # plt.rcParams.update({'font.size':20})
    ax.set_title(f"AUC results of baseline and {feature}",fontsize=24)
    ax.grid(True)
    # fill with colors
    colors = ['pink', 'lightblue']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    plt.savefig(f"{hp['root_dir']}boxplot_roc.png",dpi=500, bbox_inches = "tight")

def paired_t_test(samples_a,samples_b):
    print(stats.ttest_rel(samples_a, samples_b))
    print(np.std(samples_a))
    print(np.std(samples_b))
    
def plot_base_sub(hp,feature):

    mode = 'test'
    root1 = "/tcmldrive/hadar/from_dgx/base_snp_2/"
    root2 =  "/tcmldrive/hadar/from_dgx/snp_model_2/"
    fpr1,tpr1,auc1,aucs1 = get_mean_roc(root1)
    fpr2,tpr2,auc2,aucs2 = get_mean_roc(root2)
    fig, ax = plt.subplots() 
    plt.rcParams.update({'font.size':14})
    ax.plot(fpr1[1], tpr1[1], color='green',
                lw=1, label='Baseline (area = %0.2f)' % auc1)
    ax.fill_between(fpr1[0], tpr1[0],tpr1[1], color='g', alpha=.1)
    ax.fill_between(fpr1[0], tpr1[1],tpr1[2], color='g', alpha=.1)
    ax.plot(fpr2[1], tpr2[1], color='blue',
                lw=1, label='$BP-CNN_{CIMP}$ (area = %0.2f)' % auc2)
    ax.fill_between(fpr2[0], tpr2[0],tpr2[1], color='blue', alpha=.1)
    ax.fill_between(fpr2[0], tpr2[1],tpr2[2], color='blue', alpha=.1)
    ax.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    plt.rcParams.update({'font.size':14})
    ax.legend(loc="lower right")
    plt.rcParams.update({'font.size':18})
    ax.set_title(f"ROC Base vs {feature}")

    ax.grid(True)
    #plt.show() 
    fig.savefig("{}roc_{}_vs.png".format(hp['root_dir'],mode),dpi=500,bbox_inches="tight") 
    # roc_boxplot(aucs1,aucs2,feature)  
    paired_t_test(aucs1,aucs2)
    data_aucs = {"aucs_base": aucs1, "aucs_sub": aucs2}
    pickle.dump( data_aucs, open( f"{hp['root_dir']}aucs.p", "wb" ) )

def roc_per_patient_matlab(p,files,labels,preds):
    names = []
    pred_MSI_per_patient, true_MSI_per_patient = [],[]
    for j,name in enumerate(p.test_patients):
       # files of this patient
        indices = [i for i, x in enumerate(files) if name in x]
        # the test list might contain patient names who's label is -1
        if len(indices) == 0:
            continue
        # MSI labels for each patient
        true_MSI = (labels[indices[0]] == 1) 
        true_MSS = (labels[indices[0]] == 0)
        # equal to 1 for MSI
        true_MSI_score = int(true_MSI)
        #load patches probabilities
        preds_patient = preds[indices]
        # print(preds_patient)
        # print(sum(preds_patient==1))
        pred_MSI_score = sum(preds_patient==1)/len(preds_patient)
    
        pred_MSI_per_patient.append(pred_MSI_score.item())
        true_MSI_per_patient.append(true_MSI_score)
        names.append(name)
    data = {"name": names, "label": true_MSI_per_patient}
    path = f"{hp['root_dir']}test_labels.csv"
    df = pd.DataFrame(data)
    df.to_csv(path)
    return true_MSI_per_patient,pred_MSI_per_patient
def get_labels(names):
    df2 = pd.read_csv('test_labels.csv')
    csv_names = np.array(df2['name'])
    csv_labels = np.array(df2['label'])
    labels = []
    for name in names:
        ind = [i for i, x in enumerate(csv_names) if name==x]
        label = csv_labels[ind[0]]
        labels.append(label)
    return labels
        
    
def matlab_roc(p):
    # try:
    #     with (open(f"{hp['root_dir']}roc_out_matlab.p", "rb")) as openfile:
    #     # with (open(f"{root}roc_out_pca_{j}.p", "rb")) as openfile:
    #         data = pickle.load(openfile)
    #         labels = np.array(data['labels'])
    #         # probs_msi = np.array(data['probs'])
    #         files = np.array(data['files'])
    #         preds = np.array(data['preds'])
    # finally:
        # a = io.loadmat('classiMSSvsMSIMUT_hadar_CRC_DX.mat')
        # probs = a['predScoresExternal']
        # probs_msi = probs[:,0];
    df = pd.read_excel('kather_res.xlsx')
    names = np.array(df['PatientID'])
    probs = np.array(df['predictedScore'])
    labels = get_labels(names)
    labels = np.array(labels)
    # df2 = pd.read_csv('test_res_data.csv',header=None) 
    # files = np.array(df2[0])
    # labels = df2[1]
    # labels = np.array(labels)
    # labels[labels=='MSS'] = 0      
    # labels[labels=='MSIMUT'] = 1

    # preds = np.array(df2[2])
    # preds[preds=='MSS'] = 0      
    # preds[preds=='MSIMUT'] = 1
    # labels,probs = roc_per_patient_matlab(p,files,labels,preds)
        
        #pickle data
        # data = {"labels": labels, "preds": preds,"files": files}
        # pickle.dump( data, open( f"{hp['root_dir']}roc_out_matlab.p", "wb" ) )
    n=len(labels)
   
    inds = list(range(len(labels)))
    total_tpr =[]
    total_fpr=[]
    total_auc=[]
    np.random.seed(13)
    for i in range(500):       
        ind_i = np.random.choice(inds, size=int(np.ceil(0.95*n)),replace=True)
        labels_i = labels[ind_i]
        # files_i = files[ind_i]
        probs_msi_i = probs[ind_i]
        # preds_i = preds[ind_i]
        # labels_p,probs_p = roc_per_patient_matlab(p,files_i,labels_i,
                                                  # preds_i)
        # print(labels_p)
        names_i = names[ind_i]
        probs_i = probs[ind_i]
        lr_fpr, lr_tpr, MSI_tp_auc = compute_roc(labels_i, probs_msi_i)
                                                
        print("iter ", i, "AUC: ", MSI_tp_auc)
    
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
    #pickle data
    data = {"mean fpr": mean_fpr, "tpr low ci": tpr_lo_ci,"mean tpr": mean_tpr,"tpr high ci":tpr_hi_ci}
    pickle.dump( data, open( f"{hp['root_dir']}data_for_roc_plot_matlab.p", "wb" ) )
    auc_ = [confidence_lower,mean_auc,confidence_upper]
    print(auc_)
    plot_roc_with_ci(fpr,tpr,auc_,hp,'test')
    # print(n)
    return mean_auc
hp = hyperparams() 
# p = Prepare(hp)  
# matlab_roc(p)




#plot_mean_auc(p,'test',hp)  
plot_base_sub(hp,"CIMP") 
