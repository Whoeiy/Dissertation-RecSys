# -*- coding: utf-8 -*-
"""
Created on Fri May  7 15:02:02 2021

@author: JenniferYu
"""

import pandas as pd
import math

from surprise import KNNBasic

from time import time

import numpy as np
import vaex


from classes.Mpd import Mpd
from classes.statistics import Stat



mpd = Mpd()
stat = Stat()


# 读入数据 
print('- 读入数据 ')
pntPath_hdf5 = '../data/hdf5/real/trainset/playlist_tracks.hdf5'
trackPath_hdf5 = '../data/hdf5/real/tracks.hdf5'
    
test_pidPath_hdf5 = '../data/hdf5/real/testset/playlists_info.hdf5'
    
df_dataset = mpd.loadMpdDataDf(pntPath_hdf5)
df_tid2tname = mpd.get_tid2tname_df(trackPath_hdf5)

# print(df_dataset.head(10))


    

def get_similar_items(tid, n = 500):
    
    inner_id = algo.trainset.to_inner_iid(tid)
    # 使用get_neighbors方法得到n个最相似的电影
    neighbors = algo.get_neighbors(inner_id, k=100)
    neighbors_iid = ( algo.trainset.to_raw_iid(x) for x in neighbors )
    rec_tids = [ x for x in neighbors_iid ]
    # print('\nten movies most similar to the %s:' % item_dict[iid])
    return list(rec_tids)

# 

def rec_for_tids(list_tid, n):
    # 进行推荐
    recommendations = list()
    for tid in list_tid:
        print('- 为歌曲', tid, '进行推荐中...')
        temp = get_similar_items(tid)
        recommendations = list(temp[:n])
        print('* 歌曲推荐结果数： ', len(recommendations))
    return recommendations

def rec_for_pids(list_pid):
    
    
    
    recommendations = pd.DataFrame()
    
    
    for pid in list_pid:
        print('- 为播放列表', pid, '进行推荐中...')
        tids = df_dataset.loc[df_dataset['user']==pid, 'item'].tolist()
        n = math.ceil(500 / len(tids))
        res_all = rec_for_tids(tids, n)
        res_all = list(set(res_all))
        
        # rec = list()
        tnames = list()
        artists = list()
        tids = list()
        
        for tid in res_all:
            if tid in tids:
                continue
            
            tnames.extend(df_tid2tname.loc[df_tid2tname['tid'] == tid, 'track_name'])
            artists.extend(df_tid2tname.loc[df_tid2tname['tid'] == tid, 'artist_name'])
            tids.append(tid)
        
        # 一个播放列表的数据
        pid_temp = pd.DataFrame({'pid': pid, 'track_name':tnames, 'tid': tids, 'artist_name':artists})
        print('* 播放列表推荐结果数： ', pid_temp.shape[0])
        recommendations.append(pid_temp.head(500))
    return recommendations




mpd = Mpd()
stat = Stat()

df_dataset = mpd.loadMpdDataDf(pntPath_hdf5)
df_tid2tname = mpd.get_tid2tname_df(trackPath_hdf5)


list_pid = mpd.get_test_pid(test_pidPath_hdf5)
list_pid = list_pid[:2]

trainset = mpd.loadMpdTrainset(pntPath_hdf5)
# testset = mpd.loadMpdTestset()

# trainset = trainset.build_full_trainset()
# testset = testset.build_testset()

item_based_sim_option = {'name': 'pearson_baseline',
               'user_based': False}
# 运行时间-开始
start_time = time()
# 运行内存-开始
start_memory = stat.show_ram()


# 获取训练集，这里取数据集全部数据
# 运行时间-开始
start_time = time()
# 运行内存-开始
start_memory = stat.show_ram()
# 考虑基线评级的协同过滤算法
algo = KNNBasic(sim_option = item_based_sim_option)
algo.fit(trainset)

# 运行时间-结束
end_time = time()
# 运行时间-结束
end_memory = stat.show_ram()
# 运行时间
elapsed = end_time - start_time
print('* 模型训练时间：', elapsed)

start = time()
recommendations = pd.DataFrame()
recommendations = rec_for_pids(list_pid)

print(recommendations.shape[0])
print("* 生成推荐结果时间：",time()-start)#完成时间 6.293345212936401

# recommendations = rec_for_pids(list_pid)
print('\n- 将推荐结果写入csv文件')
# to csv
recommendations.to_csv(r'../data/result/knn/origin/recommend_50K_icf.csv')

print('\n- 将推荐结果写入hdf5文件')
# to hdf5
vaex_df = vaex.from_pandas(recommendations, copy_index=False)
vaex_df.export_hdf5('../data/result/knn/origin/recommend_50K_icf.hdf5')

print('\n=======')
print("DONE")
print('=======\n')
