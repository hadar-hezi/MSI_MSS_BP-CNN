#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 10:20:56 2021

@author: hadar.hezi@bm.technion.ac.il
"""
from skimage import io
import matplotlib.pyplot as plt



def plot_patients_hist(patient_names,k,paths,patient_dict,mode):
    """
    This function plots histograms of images.
    Each graph represents one patient. contains the histograms of all images this patient has.   

    Parameters
    ----------
    patient_names : list
        list of patient names
    k : int
        numer of patients to plot
    paths : list
        list of image files paths
    patient_dict : dict
        data structure, contains a list of paths attached to every patient
    mode : string
        can be 'train' or 'test' 

    Returns
    -------
    None.

    """
    hist_patients = patient_names[:k]
    for patient in hist_patients:
        for path in patient_dict[patient]['paths']:
            # load patient images
            image = io.imread(path)
             # take hist of image colors
            _ = plt.hist(image.ravel(), bins = 256, color = 'orange', )
        plt.title(f"histogram of patient {patient} from {mode} set")        
        # plot histograms together 
        plt.show()
    
     
