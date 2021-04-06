# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 14:48:30 2021

@author: JenniferYu
"""

from collections import defaultdict
import numpy as np
import pandas as pd
from pandas import DataFrame

class Knn:
    
    def findRecomendationsUsers(self, model, matrix, data, query_index):
        print("in function fru")
        print(matrix.head(10))
        
        distances, indices = model.kneighbors([matrix.iloc[query_index, :]], n_neighbors = 10)
        # distances, indices = model.kneighbors([matrix.iloc[query_index, :]], n_neighbors = 10)
        for i in range(0, len(distances.flatten())):
            if i == 0:
                print("Searching recommendation for user: ", matrix.index[query_index])
            else:
                rows = data.loc[data['user'] == matrix.index[indices.flatten()[i]] ]
                for item in rows.values:      
                  print("\n  User: ", item[0])
                  print("    Play Count: ", item[2])  
                  print("    Song: ",  item[1])
               