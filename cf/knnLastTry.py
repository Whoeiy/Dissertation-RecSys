#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 21:12:35 2021

@author: whoeiy
"""

import pickle
import os
import psutil
import pandas as pd

from surprise import KNNBasic
from surprise import Dataset                                                     
from surprise import Reader                                                      
from surprise.model_selection import PredefinedKFold
from surprise import dump
from surprise import accuracy
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split

from collections import defaultdict
from time import time


from classes.Mpd import Mpd
from classes.statistics import Stat

pntPath_hdf5 = '../data/hdf5/real/trainset/playlist_tracks.hdf5'
trackPath_hdf5 = '../data/hdf5/real/tracks.hdf5'

mpd = Mpd()
stat = Stat()

trainset = mpd.loadMpdTrainset(pntPath_hdf5)
# testset = mpd.loadMpdTestset()

# trainset = trainset.build_full_trainset()
# testset = testset.build_testset()

sim_options = {'name': 'cosine',
               'user_based': True}
# 运行时间-开始
start_time = time()
# 运行内存-开始
start_memory = stat.show_ram()

algo = KNNBasic(sim_options=sim_options)
# predictions = algo.test()
# rmse(predictions)

# cross_validate(algo, trainset, cv=5, verbose=True)
# dump.dump('./dump_file', predictions, algo)

trainset, testset = train_test_split(trainset, test_size=0.25)
predictions = algo.fit(trainset,).test(testset, verbose=True)
# 运行时间-结束
end_time = time()
# 运行时间-结束
end_memory = stat.show_ram()

# accuracy.rmse(predictions)

def get_Iu(uid):
    """ return the number of items rated by given user
    args: 
      uid: the id of the user
    returns: 
      the number of items rated by the user
    """
    try:
        return len(trainset.ur[trainset.to_inner_uid(uid)])
    except ValueError: # user was not part of the trainset
        return 0
    
def get_Ui(iid):
    """ return number of users that have rated given item
    args:
      iid: the raw id of the item
    returns:
      the number of users that have rated the item.
    """
    try: 
        return len(trainset.ir[trainset.to_inner_iid(iid)])
    except ValueError:
        return 0
    
df = pd.DataFrame(predictions, columns=['uid', 'iid', 'rui', 'est', 'details'])
df['Iu'] = df.uid.apply(get_Iu)
df['Ui'] = df.iid.apply(get_Ui)
df['err'] = abs(df.est - df.rui)
 
# best_predictions = df.sort_values(by='err')[:10]
# worst_predictions = df.sort_values(by='err')[-10:]

# the_predictions = df.loc[df['uid']==7000].sort_values(by='err')

# print(worst_predictions)
# print(the_predictions)


def get_top_n(predictions, n):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
 
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
 
    return top_n

topNPredicted = get_top_n(predictions, n=500)
 
#打印为每个用户推荐的10部电影和对它们的评分


    # 定义两个list用于存储文章的title和推荐分数ccccccc。
df_tid2tname = mpd.get_tid2tname_df(trackPath_hdf5)
df_result = pd.DataFrame(columns=['pid', 'track_name', 'tid', 'artist_name', 'score'])

for uid, user_ratings in topNPredicted.items():
    # df_result.loc[0] = CC
    for iid, rating in user_ratings:
        tname = df_tid2tname.loc[df_tid2tname.tid == iid].track_name
        artist = df_tid2tname.loc[df_tid2tname.tid == iid].artist_name
        df_result.loc[df_result.shape[0]] = [uid, tname, iid, artist, round(rating, 1)]
    # print(uid, [(iid,round(rating,1)) for (iid, rating) in user_ratings])
    # print(df_result.head(20))
    
    
    
    df_result.to_csv(r'../data/result/knn/recommend_100K.csv', index=None)
    print("DONE.")  
        
        
# 运行时间
elapsed = end_time - start_time
print(elapsed)


# 运行内存
print(f'一共占用{end_memory - start_memory}MB')





