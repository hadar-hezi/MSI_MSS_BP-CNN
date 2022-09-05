#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 23:48:31 2021

@author: hadar.hezi@bm.technion.ac.il
"""

from __future__ import print_function, division

import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from sklearn.metrics import roc_curve, auc, roc_auc_score
import os
import sys

from compute_roc import *



def get_auc(summary,p,hp,mode):   
    labels,probs = roc_per_patient(summary,p,hp,mode)
    fpr, tpr, MSI_auc = compute_roc(labels, probs)
    plot_roc(fpr,tpr,MSI_auc,hp,mode)

    return MSI_auc