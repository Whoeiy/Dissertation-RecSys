#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 15:34:46 2021

@author: whoeiy
"""
from classes.dataExtract_1M import extractor
# from classes.dataExtract_100K import extractor

# 原生数据 - json格式
data_json_path = "../../data_json_20K"  # 20K
# data_json_path = "../../data_json_50K"  # 50K
# data_json_path = "../../data_json_100K" # 100K
# data_json_path = "../../data_json"

'''
dataset_type - 生成数据集的形式：
    # 1 - 不划分trainset和testset
    # 2 - 划分trainset和testset
output_type - 输出存储数据集的格式：
    # 1 - csv
    # 2 - hdf5
'''
dataset_type = 2
output_type = 2

extractor(data_json_path, dataset_type, output_type)