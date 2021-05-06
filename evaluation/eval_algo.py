# -*- coding: utf-8 -*-
"""
Created on Wed May  5 11:23:15 2021

@author: JenniferYu
"""

import numpy as np
import pandas as pd
import math


def r_precision(rec, actual):
    actual = np.unique(actual)
    mask = np.in1d(rec[:len(actual)], actual)
    
    # mask = np.in1d(rec[:l], actual)
    # mask = np.in1d(rec, actual)
    res = (mask.sum() / len(actual))
    
    if math.isnan(res):
        print('r_prec is nan')
        print(mask.sum())
        print(actual)
        print(mask)
        print(res)
    if res > 1:
        print('r_prec is bigger than one')
        print(mask.sum())
        print(actual)
        print(mask)
        print(res) 
    # print(mask)
    # print(mask.sum)
    print('done')
    return res

def dcg(rec, actual, k=200):
    rel = np.in1d(rec[:k], actual) * 1
    dcg = np.sum(rel / np.log2(1 + np.arange(1, k + 1)))
    
    return dcg

def ndcg(rec, actual):
    actual = np.unique(actual)
    
    cut = np.in1d(actual, rec).sum()
    if cut == 0:
        return 0
    idcg = dcg(actual, actual, min(len(rec), len(actual)))
    rdcg = dcg(rec, actual)
    
    return rdcg / idcg