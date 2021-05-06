#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 22:31:24 2021

@author: whoeiy
"""


import pandas as pd
import vaex
import scipy.sparse as sparse
import numpy as np
import implicit
from time import time
from sklearn.preprocessing import MinMaxScaler
from classes.Mpd import Mpd 
from classes.statistics import Stat 

from concurrent.futures import ProcessPoolExecutor

playlist_vecs = list()
track_vecs = list()
sparse_playlist_track = list()

print('\n=======')
print('加权正则矩阵分解算法')
print('=======\n')




    # 为指定playlist推荐tracks。
def recommend(pid, sparse_playlist_track, playlist_vecs, track_vecs, num_tracks=500):
    #*****************得到指定playlist对所有tracks的评分向量******************************
    # 将该playlist向量乘以内容矩阵(做点积),得到该用户对所有文章的评价分数向量
    rec_vector = playlist_vecs[pid,:].dot(track_vecs.T).toarray()
    
    # **********过滤掉playlists中已经出现过的track(将其评分值置为0),因为已经出现过的track不应该被推荐*******
    # 从稀疏矩阵sparse_playlist_track中获取指定playlist对所有tracks的评价分数
    playlist_interactions = sparse_playlist_track[pid,:].toarray()
    # 为该playlist的对所有tracks的评价分数+1，那些没有出现在playlist的track分数就会等于1(原来是0)
    playlist_interactions = playlist_interactions.reshape(-1) + 1
    # 将那些已经在playlist中出现过的track的分数置为0
    playlist_interactions[playlist_interactions > 1] = 0        
    # 将该playlist的评分向量做标准化处理,将其值缩放到0到1之间
    min_max = MinMaxScaler()
    rec_vector_scaled = min_max.fit_transform(rec_vector.reshape(-1,1))[:,0]
    # 过滤掉在该playlist中出现过的tracks，这些trakcs的评分会和0相乘
    recommend_vector = playlist_interactions * rec_vector_scaled
    
    #*************将余下的评分分数排序，并输出分数最高的500首tracks************************
    track_idx = np.argsort(recommend_vector)[::-1][:num_tracks]
    
    # 定义4个list用于存储track的信息
    
    # print(df_tid2tname.head(10))
    tnames = list()
    artists = list()
    tids = list()
    scores = list()
 
    for idx in track_idx:
        # 将4个字段添加到list中
        tnames.extend(df_tid2tname.loc[df_tid2tname['tid'] == idx, 'track_name'])
        artists.extend(df_tid2tname.loc[df_tid2tname['tid'] == idx, 'artist_name'])
        tids.append(idx)
        scores.append(recommend_vector[idx])
 
    recommendations = pd.DataFrame({'pid': pid, 'track_name':tnames, 'tid': tids, 'artist_name':artists, 'score': scores})
 
    return recommendations


def rec_for_pids(list_pid):
    # 进行推荐
    recommendations = pd.DataFrame()
    for pid in list_pid:
        print('为播放列表', pid, '进行推荐中...')
        temp = recommend(pid, sparse_playlist_track, playlist_vecs, track_vecs)
        recommendations = recommendations.append(temp)
        print(recommendations.shape[0])
    return recommendations


mpd = Mpd()
stat = Stat()
    
pntPath_hdf5 = '../data/hdf5/real/trainset/playlist_tracks.hdf5'
trackPath_hdf5 = '../data/hdf5/real/tracks.hdf5'
    
test_pidPath_hdf5 = '../data/hdf5/real/testset/playlists_info.hdf5'
    
df_dataset = mpd.loadMpdDataDf(pntPath_hdf5)
df_tid2tname = mpd.get_tid2tname_df(trackPath_hdf5)
    
sparse_track_playlist = sparse.csr_matrix((df_dataset['rating'].astype(float), (df_dataset['item'], df_dataset['user'])))
# 评分矩阵
sparse_playlist_track = sparse.csr_matrix((df_dataset['rating'].astype(float), (df_dataset['user'], df_dataset['item'])))
    
# print(sparse_track_playlist.shape)
# print(sparse_playlist_track.shape)
    
# 运行时间-开始
start_time = time()
# 运行内存-开始
start_memory = stat.show_ram()
    
# 训练模型
#  设置20个特征因子
alpha = 40
data = (sparse_track_playlist * alpha).astype('double')
    
print("- 模型训练中...")
model = implicit.als.AlternatingLeastSquares(factors=20, regularization=0.1, iterations=50)
model.fit(data)

# 计算track之间的相似度
# tid = 0
# n_similar = 10

#获取playlist矩阵
playlist_vecs = model.user_factors
#获取track矩阵
track_vecs = model.item_factors
print('* playlist隐含特征向量:', playlist_vecs.shape)
print('* track隐含特征向量:', track_vecs.shape)
# track_norms = np.sqrt((track_vecs * track_vecs).sum(axis=1))
#计算指定的tid与其他所有tracks的相似度
# scores = track_vecs.dot(track_vecs[tid]) / track_norms
#获取相似度最大的10首track
# top_idx = np.argpartition(scores, -n_similar)[-n_similar:]
#组成content_id和title的元组
# similar = sorted(zip(top_idx, scores[top_idx] / track_norms[tid]), key=lambda x: -x[1])

# 运行时间-结束
end_time = time()
# 运行时间-结束
end_memory = stat.show_ram()
# 运行时间
elapsed = end_time - start_time
print('* 模型训练时间：', elapsed)



# for track in similar:
    # print(track)


# 为playlist推荐track

# 从model中获取经过训练的用户和内容矩阵,并将它们存储为稀疏矩阵
playlist_vecs = sparse.csr_matrix(model.user_factors)
track_vecs = sparse.csr_matrix(model.item_factors)

list_pid = mpd.get_test_pid(test_pidPath_hdf5)
print('* 播放列表数（测试集）: ', len(list_pid))

# list_pid = list_pid[500:1000]
    
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
    recommendations.to_csv(r'../data/result/implicit/origin/recommend_100K_200.csv')
    
    print('\n- 将推荐结果写入hdf5文件')
    # to hdf5
    vaex_df = vaex.from_pandas(recommendations, copy_index=False)
    vaex_df.export_hdf5('../data/result/implicit/origin/recommend_100K_200.hdf5')
    
    print('\n=======')
    print("DONE")
    print('=======\n')




