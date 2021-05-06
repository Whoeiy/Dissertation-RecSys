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
