# -*- coding: utf-8 -*-
"""
Created on Thu May  6 17:29:11 2021

@author: JenniferYu
"""

import pandas as pd

df = pd.DataFrame(['a', 'b', 'c'])
df_list = df.index.tolist()
print(df_list)
index = [0,1]
print(df.loc[[i for i in index]])