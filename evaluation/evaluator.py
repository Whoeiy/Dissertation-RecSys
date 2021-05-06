# -*- coding: utf-8 -*-
"""
Created on Wed May  5 11:22:48 2021

@author: JenniferYu
"""

from classes.Mpd import Mpd
import eval_algo as eval
import math



rec_res = '../data/result/implicit/recommend_100K_100.hdf5'
true_res = '../data/hdf5/real/testset/playlist_tracks_true.hdf5'

test_pidPath_hdf5 = '../data/hdf5/real/testset/playlists_info.hdf5'

mpd = Mpd()

n = 100
# get res df
df_rec = mpd.get_res_df(rec_res)
df_true = mpd.get_res_df(true_res)
# get pid list
list_pid = mpd.get_test_pid(test_pidPath_hdf5)
list_pid = list_pid[:n]


rprec = list()
for pid in list_pid:
    rec_tid = df_rec.loc[df_rec['pid'] == pid, 'tid'].tolist()
    true_tid = df_true.loc[df_true['pid'] == pid, 'tid'].tolist()
    rp_res = eval.r_precision(rec_tid, true_tid)
    if math.isnan(rp_res):
        rp_res = 0.0
    print(pid, " r-precision: ", rp_res)
    rprec.append(rp_res)
    
all_rprec = sum(rprec) / n


ndcg = list()
for pid in list_pid:
    rec_tid = df_rec.loc[df_rec['pid'] == pid, 'tid'].tolist()
    true_tid = df_true.loc[df_true['pid'] == pid, 'tid'].tolist()
    ndcg_res = eval.ndcg(rec_tid, true_tid)
    print(ndcg_res)
    ndcg.append(ndcg_res)

all_ndcg = sum(ndcg) / n
print('\nndcg', all_ndcg)

print('\nr-precision:', all_rprec)

    
   