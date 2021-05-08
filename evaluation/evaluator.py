# -*- coding: utf-8 -*-
"""
Created on Wed May  5 11:22:48 2021

@author: JenniferYu
"""

from classes.Mpd import Mpd
from classes.getData import getData
import eval_algo as eval
import math



# rec_res = '../data/result/implicit/origin/recommend_100K_200.hdf5'
rec_res = '../data/result/knn/origin/recommend_50K_ucf.hdf5'
true_res = '../data/hdf5/real/testset/playlist_tracks_true.hdf5'

test_pidPath_hdf5 = '../data/hdf5/real/testset/playlists_info.hdf5'
tracksPath_hdf5 = '../data/hdf5/real/tracks.hdf5'

mpd = Mpd()

n = 70
# get res df
df_rec = mpd.get_res_df(rec_res)
# rec_res = '../data/result/implicit/origin/recommend_100K_500.hdf5'
# df_rec = df_rec.append(mpd.get_res_df(rec_res))

df_true = mpd.get_res_df(true_res)
# get pid list
list_pid = mpd.get_test_pid(test_pidPath_hdf5)
list_pid = list_pid[:n]

df_track = getData().getTracks(tracksPath_hdf5)
df_test_p = df_track.sample(n=500, replace=False, random_state=1)
tid_list = df_test_p['tid'].tolist()
print(len(tid_list))
# print(tid_list[:100])


rprec = list()
for pid in list_pid:
    rec_tid = df_rec.loc[df_rec['pid'] == pid, 'tid'].tolist()
    true_tid = df_true.loc[df_true['pid'] == pid, 'tid'].tolist()
    if pid == 107970:
        print(pid, 'rec_tid', rec_tid)
        print(pid, 'true_tid', true_tid)
    # rp_res = eval.r_precision(rec_tid, true_tid)
    # print(rec_tid)
    df_test_p = df_track.sample(n=500, replace=False, random_state=1)
    tid_list = df_test_p['tid'].tolist()
    rp_res = eval.r_precision(tid_list, true_tid)
    if math.isnan(rp_res):
        rp_res = 0.0
    print(pid, " r-precision: ", rp_res)
    rprec.append(rp_res)
    
all_rprec = sum(rprec) / n


ndcg = list()
for pid in list_pid:
    rec_tid = df_rec.loc[df_rec['pid'] == pid, 'tid'].tolist()
    true_tid = df_true.loc[df_true['pid'] == pid, 'tid'].tolist()
    # ndcg_res = eval.ndcg(rec_tid, true_tid)
    df_test_p = df_track.sample(n=500, replace=False, random_state=1)
    tid_list = df_test_p['tid'].tolist()
    ndcg_res = eval.ndcg(tid_list, true_tid)
    print(ndcg_res)
    ndcg.append(ndcg_res)

all_ndcg = sum(ndcg) / n

print('\n======')
print('评估(evaluation)：WRMF')
print('======\n')
print('* 测试集播放列表数: ', n)
print('* 准确度(precision): ', all_rprec)
print('* NDCG: ', all_ndcg)
# print('\nndcg', all_ndcg)

# print('\nr-precision:', all_rprec)
print('\n=======')
print("DONE")
print('=======\n')
# print(df_rec.shape[0])

    
   