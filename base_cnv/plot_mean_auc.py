from prepare_data import Prepare
from log_file import *
from hyperparams import hyperparams
from compute_roc import *
import pickle
import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats

    
def plot_mean_auc(p,mode,hp):
    total_tpr =[]
    total_fpr=[]
    total_auc=[]
    # TODO:
    # do a shuffle on the indices and take 95% of then at each iteration
    for j in range(hp['n_folds']):
        try:
            with (open(f"roc_out_lr_{j}.p", "rb")) as openfile:
            # with (open(f"{root}roc_out_pca_{j}.p", "rb")) as openfile:
                data = pickle.load(openfile)
        except:
            summary = read_results(f"{hp['test_res_file']}_{j}.csv")
            labels,probs = roc_per_patient(summary,p,hp,mode)
            
            #pickle data
            data = {"labels": labels, "probs": probs}
            pickle.dump( data, open( f"roc_out_lr_{j}.p", "wb" ) )
        # prec, recall, thresholds = precision_recall_curve(labels,probs)
        # plot_prec(recall,prec, f1,hp,mode)
        labels = data['labels']
        probs = data['probs']
        # patient_names = p.test_patients
        # path_atlas = f"clinical_Atlas.csv"
        # if os.path.isfile(path_atlas):
            # df = pd.read_csv(path_atlas)     
            # atlas_names = list(df['TCGA Participant Barcode'])
            # lr_label = list(df['Left or right colon'])
        # for i,name in enumerate(patient_names):
            # for k,df_name in enumerate(atlas_names):
                # if df_name == name:
                    # if  lr_label[k] == 'Left':                  
                        # prior = 0.1
                    # elif  lr_label[k] == 'Right':  
                        # prior=0.9                    
                        # # if probs[i] >0.2:                    
                            # # prior = 1.5
                        # # else:
                            # # prior=1
                    # else:
                        # prior = 1
                    # probs[i] = probs[i]*prior
        # data = {"labels": labels, "probs": probs}
        # pickle.dump( data, open( f"roc_out_lr_{j}.p", "wb" ) )
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
    #pickle.dump( data, open( f"{hp['root_dir']}data_for_snp_lr_roc.p", "wb" ) )
    auc_ = [confidence_lower,mean_auc,confidence_upper]
    print(auc_)
    std=np.std(total_auc)
    plot_roc_with_ci(fpr,tpr,auc_,hp,mode,std)
    return mean_auc, fpr,tpr

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
    #ax.legend(["base std: ", "feature std: "])
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
    
    
def get_mean_roc(root):    
    total_tpr =[]
    total_fpr=[]
    total_auc=[]
    for j in range(5):
        with (open(f"{root}_{j}.p", "rb")) as openfile:
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

def plot_base_sub(hp,feature):

    mode = 'test'
    #root1 = f"/tcmldrive/hadar/MSI_MSS_project/hist_eq_model/base_for_cnv/base_for_cnv1/"
    #root2 = f"/tcmldrive/hadar/MSI_MSS_project/base_cnv/cnv_res/"
    root1 = "roc_out"
    root2 = "roc_out_lr"
    fpr1,tpr1,auc1,aucs1 = get_mean_roc(root1)
    fpr2,tpr2,auc2,aucs2 = get_mean_roc(root2)
    fig, ax = plt.subplots() 
    print(plt.rcParams.get('figure.figsize'))
    plt.rcParams.update({'font.size':14})
    ax.plot(fpr1[1], tpr1[1], color='green',
                lw=1, label='Baseline (area = %0.2f)' % auc1)
    ax.fill_between(fpr1[0], tpr1[0],tpr1[1], color='g', alpha=.1)
    ax.fill_between(fpr1[0], tpr1[1],tpr1[2], color='g', alpha=.1)
    ax.plot(fpr2[1], tpr2[1], color='blue',
                lw=1, label='Side prior (area = %0.2f)' % auc2)
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
    fig.savefig("{}roc_{}.png".format("",mode),dpi=500,bbox_inches="tight")   
    #roc_boxplot(aucs1,aucs2,feature)  
    data_aucs = {"aucs_base": aucs1, "aucs_sub": aucs2}
    #pickle.dump( data_aucs, open( f"{hp['root_dir']}aucs.p", "wb" ) )
    paired_t_test(aucs1,aucs2)
hp = hyperparams()
p = Prepare(hp)   
plot_mean_auc(p,'test',hp)  
#plot_base_sub(hp,"Side prior") 