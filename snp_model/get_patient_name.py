#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 18:42:03 2021

@author: hadar.hezi@bm.technion.ac.il
"""
import re
import numpy as np


def get_patient_name(file):
    # to get patient name take dash - number 3
    # to get slide take dash number 5
    dash_num =3
    start_ind = file.find("blk-")
    if start_ind > -1:      
        file = file[start_ind:]
    start_ind = file.find("-TCGA-")
    name = file[start_ind+1:]
    # dashes_ind= np.nonzero(name=='-')
    # print("dashes ", len(dashes_ind))
    dashes_ind = [m.start() for m in re.finditer('-', name)]
    end_ind = dashes_ind[dash_num-1]
    name = name[:end_ind]
    return name