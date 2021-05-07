# -*- coding: utf-8 -*-
"""
Created on Thu May  6 15:34:03 2021

@author: JenniferYu
"""

from classes.Mpd import Mpd

mpd = Mpd()

true_res = '../data/hdf5/real/testset/playlists_info.hdf5'
df = mpd.get_test_playlists(true_res)

# str = df.loc[df['pid'] == 107970]
# str = df.loc[df['test_type'] == 100.0]
str = df.loc[0]
print(str)

'''
rec_res1 = '../data/result/implicit/origin/recommend_100K_0.hdf5'
rec_res = '../data/result/implicit/origin/recommend_100K_500.hdf5'
true_res = '../data/hdf5/real/testset/playlist_tracks_true.hdf5'
df = mpd.get_res_df(true_res)
temp = df.loc[df['pid'] == 107970]
print(temp)
'''

pntPath_hdf5 = '../data/hdf5/real/trainset/playlist_tracks.hdf5'
trackPath_hdf5 = '../data/hdf5/real/tracks.hdf5'
    
test_pidPath_hdf5 = '../data/hdf5/real/testset/playlists_info.hdf5'
    
df_dataset = mpd.loadMpdDataDf(pntPath_hdf5)
print(df_dataset.loc[df_dataset['item']==71215])
print(df_dataset.loc[df_dataset['user']==3])

list_pid = mpd.get_test_pid(test_pidPath_hdf5)
print(list_pid[:5])


tids = df_dataset.loc[df_dataset['user']==195, 'item'].tolist()

print(tids)
temp = df_dataset['item'].unique().tolist()
print(len(temp))