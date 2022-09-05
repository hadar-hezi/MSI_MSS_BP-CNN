#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 21:13:32 2021

@author: hadar.hezi@bm.technion.ac.il
"""
import statistics as st
import numpy as np
from scipy import stats

# def paired_t_test(samples_a,samples_b):
#     mean_a = st.mean(samples_a)
#     mean_b= st.mean(samples_b)
#     diff = samples_a - samples_b
#     n = len(samples_b)
#     s = st.stdev(diff)
#     t_res = pow(n,0.5)*(mean_a-mean_b)/s
#     print(t_res)
   
samples_a = np.array([0.80307,0.79625,0.6859,0.73336,0.79158])
samples_b = np.array([0.82484,0.8659,0.82828,0.79677,0.80925])
print(stats.ttest_rel(samples_a, samples_b))
print(np.std(samples_a))
print(np.std(samples_b))
# paired_t_test(samples_a,samples_b)