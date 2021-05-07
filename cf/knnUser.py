#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 21:12:35 2021

@author: whoeiy
"""

import pandas as pd

from surprise import KNNBaseline

from time import time

import numpy as np
import vaex


from classes.Mpd import Mpd
from classes.statistics import Stat

pntPath_hdf5 = '../data/hdf5/real/trainset/playlist_tracks.hdf5'
trackPath_hdf5 = '../data/hdf5/real/tracks.hdf5'
test_pidPath_hdf5 = '../data/hdf5/real/testset/playlists_info.hdf5'



pntPath_hdf5 = '../data/hdf5/real/trainset/playlist_tracks.hdf5'
trackPath_hdf5 = '../data/hdf5/real/tracks.hdf5'
    
test_pidPath_hdf5 = '../data/hdf5/real/testset/playlists_info.hdf5'

print('\n=======')
print('基于近邻的协同过滤算法（User-based）')
print('=======\n')


def get_similar_users_recommendations(pid, n=500):
    # 将原始id转换为内部id
    inner_id = algo.trainset.to_inner_uid(pid)
    # 使用get_neighbors方法得到10个最相似的用户
    neighbors = algo.get_neighbors(inner_id, k=100)
    neighbors_uid = ( algo.trainset.to_raw_uid(x) for x in neighbors )

    rec = list()
    tnames = list()
    artists = list()
    tids = list()
    
    
    cur_tid = df_dataset.loc[df_dataset['user']==pid, 'item'].tolist()
    #把评分为5的电影加入推荐列表
    for user in neighbors_uid:
        if len(rec) > n:
            break
        item = df_dataset[df_dataset['user']==user]
        item = item['item']
        for i in item:
            if i in cur_tid:
                continue
            if i in rec:
                continue
            rec.append(i)
            
            tnames.extend(df_tid2tname.loc[df_tid2tname['tid'] == i, 'track_name'])
            artists.extend(df_tid2tname.loc[df_tid2tname['tid'] == i, 'artist_name'])
            tids.append(i)
            
    recommendations = pd.DataFrame({'pid': pid, 'track_name':tnames, 'tid': tids, 'artist_name':artists})
    return recommendations.head(n)



def rec_for_pids(list_pid):
    # 进行推荐
    recommendations = pd.DataFrame()
    for pid in list_pid:
        print('为播放列表', pid, '进行推荐中...')
        temp = get_similar_users_recommendations(pid)
        recommendations = recommendations.append(temp)
        print('推荐结果数： ', recommendations.shape[0])
    return recommendations


mpd = Mpd()
stat = Stat()

df_dataset = mpd.loadMpdDataDf(pntPath_hdf5)
df_tid2tname = mpd.get_tid2tname_df(trackPath_hdf5)


list_pid = mpd.get_test_pid(test_pidPath_hdf5)

trainset = mpd.loadMpdTrainset(pntPath_hdf5)
# testset = mpd.loadMpdTestset()

# trainset = trainset.build_full_trainset()
# testset = testset.build_testset()

user_based_sim_option = {'name': 'cosine',
               'user_based': True}

item_based_sim_option = {'name': 'cosine',
               'user_based': True}
# 运行时间-开始
start_time = time()
# 运行内存-开始
start_memory = stat.show_ram()


# 获取训练集，这里取数据集全部数据
# 运行时间-开始
start_time = time()
# 运行内存-开始
start_memory = stat.show_ram()
print("- 模型训练中...")
# 考虑基线评级的协同过滤算法
algo = KNNBaseline(sim_option = user_based_sim_option)
# 拟合训练集
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
recommendations.to_csv(r'../data/result/knn/origin/recommend_50K_ucf.csv')

print('\n- 将推荐结果写入hdf5文件')
# to hdf5
vaex_df = vaex.from_pandas(recommendations, copy_index=False)
vaex_df.export_hdf5('../data/result/knn/origin/recommend_50K_ucf.hdf5')

print('\n=======')
print("DONE")
print('=======\n')


'''
if __name__ == '__main__':
    
    threads = 4 
    print('\n- 多进程生成推荐结果，进程数: ', threads)
    # pool = ProcessPoolExecutor(threads)
    # res = pool.map(rec_for_pids, np.array_split(list_pid, threads))
    # pool.shutdown()
    
    recommendations = pd.DataFrame()
    # for r in res:
    
    p = ProcessPoolExecutor(threads)
    start = time()
    arrs = np.array_split(list_pid, threads)
    for arr in arrs:
        future = p.submit(rec_for_pids, arr)
        print("- 进程结束")
        # print(future.result())
        recommendations = recommendations.append(future.result())
    
    
    p.shutdown(wait=True)
    print(recommendations.shape[0])
    print("* 生成推荐结果时间：",time()-start)#完成时间 6.293345212936401
    
    # recommendations = rec_for_pids(list_pid)
    print('\n- 将推荐结果写入csv文件')
    # to csv
    recommendations.to_csv(r'../data/result/knn/origin/recommend_50K_ucf.csv')
    
    print('\n- 将推荐结果写入hdf5文件')
    # to hdf5
    vaex_df = vaex.from_pandas(recommendations, copy_index=False)
    vaex_df.export_hdf5('../data/result/knn/origin/recommend_50K_ucf.hdf5')
    
    print('\n=======')
    print("DONE")
    print('=======\n')





'''



