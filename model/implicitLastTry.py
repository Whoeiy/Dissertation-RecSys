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
import random
import implicit
from time import time
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from classes.Mpd import Mpd 
from classes.statistics import Stat 

from concurrent.futures import ProcessPoolExecutor

playlist_vecs = list()
track_vecs = list()
sparse_playlist_track = list()


    # 为指定用户推荐文章。
def recommend(pid, sparse_playlist_track, playlist_vecs, track_vecs, num_tracks=500):
    #*****************得到指定用户对所有文章的评分向量******************************
    # 将该用户向量乘以内容矩阵(做点积),得到该用户对所有文章的评价分数向量
    rec_vector = playlist_vecs[pid,:].dot(track_vecs.T).toarray()
    
    #**********过滤掉用户已经评分过的文章(将其评分值置为0),因为用户已经发生过交互行为的文章不应该被推荐*******
    # 从稀疏矩阵sparse_person_content中获取指定用户对所有文章的评价分数
    playlist_interactions = sparse_playlist_track[pid,:].toarray()
    # 为该用户的对所有文章的评价分数+1，那些没有被该用户看过(view)的文章的分数就会等于1(原来是0)
    playlist_interactions = playlist_interactions.reshape(-1) + 1
    # 将那些已经被该用户看过的文章的分数置为0
    playlist_interactions[playlist_interactions > 1] = 0        
    # 将该用户的评分向量做标准化处理,将其值缩放到0到1之间。
    min_max = MinMaxScaler()
    rec_vector_scaled = min_max.fit_transform(rec_vector.reshape(-1,1))[:,0]
    # 过滤掉和该用户已经交互过的文章，这些文章的评分会和0相乘。
    recommend_vector = playlist_interactions * rec_vector_scaled
    
    #*************将余下的评分分数排序，并输出分数最大的10篇文章************************
    # 根据评分值进行排序,并获取指定数量的评分值最高的文章
    track_idx = np.argsort(recommend_vector)[::-1][:num_tracks]
    
    # 定义两个list用于存储文章的title和推荐分数ccccccc。
    
    # print(df_tid2tname.head(10))
    tnames = list()
    artists = list()
    tids = list()
    scores = list()
 
    for idx in track_idx:
        # 将title和分数添加到list中
        tnames.extend(df_tid2tname.loc[df_tid2tname['tid'] == idx, 'track_name'])
        artists.extend(df_tid2tname.loc[df_tid2tname['tid'] == idx, 'artist_name'])
        tids.append(idx)
        scores.append(recommend_vector[idx])
 
    recommendations = pd.DataFrame({'pid': pid, 'track_name':tnames, 'tid': tids, 'artist_name':artists, 'score': scores})
 
    return recommendations


def rec_for_pids(list_pid):
    
    
    recommendations = pd.DataFrame()
    for pid in list_pid:
        print('recommending for pid ', pid, '...')
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
sparse_playlist_track = sparse.csr_matrix((df_dataset['rating'].astype(float), (df_dataset['user'], df_dataset['item'])))
    
print(sparse_track_playlist.shape)
print(sparse_playlist_track.shape)
    
# 运行时间-开始
start_time = time()
# 运行内存-开始
start_memory = stat.show_ram()
    
# 训练模型
#  设置20个特征因子
alpha = 40
data = (sparse_track_playlist * alpha).astype('double')
    
print("model training...")
model = implicit.als.AlternatingLeastSquares(factors=20, regularization=0.1, iterations=50)
model.fit(data)

# 计算track之间的相似度
tid = 0
n_similar = 10

#获取用户矩阵
playlist_vecs = model.user_factors
#获取内容矩阵
track_vecs = model.item_factors
track_norms = np.sqrt((track_vecs * track_vecs).sum(axis=1))
#计算指定的content_id 与其他所有文章的相似度
scores = track_vecs.dot(track_vecs[tid]) / track_norms
#获取相似度最大的10篇文章
top_idx = np.argpartition(scores, -n_similar)[-n_similar:]
#组成content_id和title的元组
similar = sorted(zip(top_idx, scores[top_idx] / track_norms[tid]), key=lambda x: -x[1])

# 运行时间-结束
end_time = time()
# 运行时间-结束
end_memory = stat.show_ram()

print(playlist_vecs.shape)
print(track_vecs.shape)

# for track in similar:
    # print(track)


# 为playlist推荐track

# 从model中获取经过训练的用户和内容矩阵,并将它们存储为稀疏矩阵
playlist_vecs = sparse.csr_matrix(model.user_factors)
track_vecs = sparse.csr_matrix(model.item_factors)

'''
pid = 404
recommendations = recommend(pid, sparse_playlist_track, playlist_vecs, track_vecs)
print(recommendations)
'''

list_pid = mpd.get_test_pid(test_pidPath_hdf5)
print(len(list_pid))

list_pid = list_pid[:100]
    




# 运行时间
elapsed = end_time - start_time
print(elapsed)



if __name__ == '__main__':
    print('multiple threads...')
    threads = 4 
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
        print("process done")
        # print(future.result())
        recommendations = recommendations.append(future.result())
    
    
    p.shutdown(wait=True)
    print(recommendations.shape[0])
    print("完成时间",time()-start)#完成时间 6.293345212936401
    
    # recommendations = rec_for_pids(list_pid)
    # to csv
    # recommendations.to_csv(r'../data/result/implicit/recommend_20K.csv')
    
    # to hdf5
    vaex_df = vaex.from_pandas(recommendations, copy_index=False)
    vaex_df.export_hdf5('../data/result/implicit/recommend_100K_100.hdf5')
    print("DONE.")
    
    
    # 运行内存
    print(f'一共占用{end_memory - start_memory}MB')




