# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 10:14:23 2021

@author: JenniferYu
"""
import os
import pandas as pd
import pandas as pd
from classes.Mpd import Mpd
from classes.Knn import Knn

mpd = Mpd()
# data = mpd.loadMpdLatest()

df_pnt = mpd.getPntDf()

print(df_pnt.head(20))

# data = Dataset.load_builtin()
# ratings = data.raw_ratings

# print(data)

# clf = BaselineOnly()
# cross_validate(clf, data, measures=['MAE'], cv=5, verbose=True)

# sim_options = {
    # 'name': 'MSD',
    # 'user_based': 'True'    
# }

# clf = KNNBasic(sim_options=sim_options)
# cross_validate(clf, data, measures=['MAE'], cv=5, verbose=True)


# from scipy.sparse import csr_matrix
# pivot rating into movie features
df_track_feature = df_pnt.pivot(
    index='user',
    columns='item',
    values='rating'
).fillna(0)

# matrix_track_features = csr_matrix(df_track_feature.values)
print(df_track_feature.head(20))


from sklearn.neighbors import NearestNeighbors
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
model_knn.fit(df_track_feature)

'''
def findRecomendationsUsers(model, matrix, data, query_index):
    distances, indices = model.kneighbors([matrix.iloc[query_index, :]], n_neighbors = 10)
    for i in range(0, len(distances.flatten())):
        if i == 0:
            print("Searching recommendation for user: ", matrix.index[query_index])
        else:
            
            rows = data.loc[data['user'] == matrix.index[indices.flatten()[i]] ]
            for item in rows.values:      
              print("\n  User: ", item[0])
              print("    Play Count: ", item[2])  
              print("    Song: ",  item[1])
 '''             
knn = Knn()

knn.findRecomendationsUsers(model_knn, df_track_feature, df_pnt, 0)
'''
            print("else")
num_playlists = len(df_pnt.pid.unique)
num_tracks = len(df_pnt.tid.unique())
print('There are {} unique playlists and {} unique tracks in this data set'.format(num_palylist, num_tracks))

'''





























