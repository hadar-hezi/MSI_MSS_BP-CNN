#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 22:28:58 2021

@author: hadar.hezi@bm.technion.ac.il
"""

import numpy as np

def hyperparams():
    hp = dict(
        batch_size=64,validation_ratio=0.3, lr=1e-4, eps=1e-8, num_workers=2,num_classes=2
    )

    hp['num_epochs'] = 15
    hp['newseed'] = 42
    hp['n_folds'] = 5
    hp['curr_fold'] = 0                   
    hp['train_res_file'] = 'train_res_data_hist_eq'
    hp['valid_res_file'] = 'valid_res_data_hist_eq'
    hp['test_res_file'] = 'test_res_data_hist_eq'
    hp['checkpoint_save'] = 'model_hist_eq'
    hp['optimizer_type'] = 'Adam'
    hp['model_type'] = 'inception'
    hp['hist_type'] = 'rgb'
    hp['pretrained'] = True
    hp['unfixed'] = False
    hp['N_boot'] = 1
    hp['experiment'] = 'MSI MSS base'
    # hp['root_dir']='MSI_MSS_project/from_dgx/base_results_for_snp/'
    # hp['data_dir'] = 'data'
    hp['root_dir']='/tcmldrive/hadar/from_dgx/base_snp/'
    hp['data_dir'] ='/tcmldrive/databases/Public/TCGA/data'
    # ========================
    return hp
