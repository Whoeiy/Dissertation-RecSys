#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 15:34:46 2021

@author: whoeiy
"""
from classes.dataExtract import extractor

data_json_path = "../../data_json_500K"

extractor(data_json_path, 1, 2)
