#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 22:31:24 2021

@author: whoeiy
"""


import pandas as pd
import scipy.sparse as sparse
import numpy as np
import random
import implicit
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from classes.Mpd import Mpd  

mpd = Mpd()

df_dataset = mpd.loadMpdDataDf()

sparse_track_playlist = sparse.csr_matrix((df_dataset['rating'].astype(float), (df_dataset['item'], df_dataset['user'])))
sparse_playlist_track = sparse.csr_matrix((df_dataset['rating'].astype(float), (df_dataset['user'], df_dataset['item'])))

print(sparse_track_playlist.shape)
print(sparse_playlist_track.shape)

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
 
print(playlist_vecs.shape)
print(track_vecs.shape)

# for track in similar:
    # print(track)


# 为playlist推荐track

def recommend(pid, sparse_playlist_track, playlist_vecs, track_vecs, num_tracks=20):
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
    df_tid2tname = mpd.get_tid2tname_df()
    tnames = []
    tids = []
    scores = []
 
    for idx in track_idx:
        # 将title和分数添加到list中
        tnames.append(df_tid2tname.loc[df_tid2tname.tid == idx].track_name)
        tids.append(idx)
        scores.append(recommend_vector[idx])
 
    recommendations = pd.DataFrame({'track_name':tnames, 'tid': tids, 'score': scores})
 
    return recommendations




# 从model中获取经过训练的用户和内容矩阵,并将它们存储为稀疏矩阵
playlist_vecs = sparse.csr_matrix(model.user_factors)
track_vecs = sparse.csr_matrix(model.item_factors)

# 为指定用户推荐文章。
pid = 7000
recommendations = recommend(pid, sparse_playlist_track, playlist_vecs, track_vecs)
print(recommendations)


recommendations.to_csv(r'../data/result/implicit/recommend_20K.csv', index=None)
print("DONE.")



