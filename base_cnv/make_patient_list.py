#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 11:03:05 2020

@author: hadar.hezi@bm.technion.ac.il

"""

import torch
import numpy as np
import pandas as pd
import os

from get_patient_name import get_patient_name
from hyperparams import hyperparams

def make_patient_list(source,save_path):
    all_names = []    
    # make list of patient names
    for dir_ in os.listdir(source):
        for file in os.listdir(os.path.join(source,dir_)):
            name = get_patient_name(file)
            all_names.append(name)
    
    # unique names list
    patient_names = list(set(all_names)) 
    patient_names.sort()
    print("number of patients: ", len(patient_names))
    data_dict = {'name':patient_names}
    df = pd.DataFrame(data_dict)
    df.to_csv(save_path)

hp = hyperparams()
# training set patients
make_patient_list(os.path.join(hp['data_dir'],'train'),f"{hp['root_dir']}train_patients.csv")
# test set patients
make_patient_list(os.path.join(hp['data_dir'],'test'),f"{hp['root_dir']}test_patients.csv")