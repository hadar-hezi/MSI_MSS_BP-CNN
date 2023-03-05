#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 22:28:58 2021

@author: hadar.hezi@bm.technion.ac.il
"""

import numpy as np

def hyperparams():
    hp = dict(
        batch_size=64, lr=1e-4, eps=1e-8, num_workers=2,num_classes=3
    )

    hp['newseed'] = 42
    hp['num_epochs'] = 15
    hp['n_folds'] = 5
    hp['curr_fold'] = 0
    hp['train_res_file'] = 'train_res_data_cnv'
    hp['valid_res_file'] = 'valid_res_data_cnv'
    hp['test_res_file'] = 'test_res_data_cnv1'
    hp['checkpoint_save'] = 'model_cnv'
    hp['optimizer_type'] = 'Adam'
    hp['model_type'] = 'inception'
    hp['pretrained'] = True
    hp['unfixed'] = False
    hp['experiment'] = 'MSI MSS cnv'
    hp['cnv_threshold'] = 0.005
    # hp['root_dir']='MSI_MSS_project/base_cnv/'
    # hp['data_dir'] = 'data'
    hp['root_dir']='/tcmldrive/hadar/MSI_MSS_project/base_cnv/cnv_res/'
    hp['data_dir'] ='/tcmldrive/databases/Public/TCGA/data'
    # ========================
    return hp
