#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 10:43:35 2022

@author: hadar.hezi@bm.technion.ac.il
"""
import numpy as np
import matplotlib.pyplot as plt
from hyperparams import *
from matplotlib import rcParams

hp = hyperparams()

roc_base = [80.301,79.625,68.659,73.336,79.158]
roc_snp = [82.484,86.59,82.848,79.677,80.925]


data = [roc_base,roc_snp]

print("medians: ", np.median(roc_base),np.median(roc_snp))
labels = ["baseline","snp"]

plt.rcParams['xtick.labelsize']='large'
# fig= plt.figure(figsize =(10, 7))
fig= plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
plt.rcParams.update({'font.size':12})
# rectangular box plot
bplot = ax.boxplot(data,
                     vert=True, 
                     showmeans=True,# vertical box alignment
                     patch_artist=True,  # fill with color
                     labels=labels)  # will be used to label x-ticks
ax.set_title("AUC results of baseline and SNP model",fontsize=20)
ax.grid(True)
plt.yticks(np.arange(68,87,2))
# fill with colors
colors = ['pink', 'lightblue']
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)
plt.savefig(f"{hp['root_dir']}boxplot_roc.png",dpi=300, bbox_inches = "tight")