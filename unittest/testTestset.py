# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 23:09:27 2021

@author: JenniferYu
"""

from classes.dataExtract import extractor

# data_json_path = "../../data_json"
data_json_path = "../../data_json_small"
dataset_type = 2
output_type = 1
extractor(data_json_path, dataset_type, output_type)
