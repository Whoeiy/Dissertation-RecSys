#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 21:12:35 2021

@author: whoeiy
"""

import pickle
import os

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

from classes.Mpd import Mpd

mpd = Mpd()

trainset = mpd.loadMpdTrainset()
# testset = mpd.loadMpdTestset()

# trainset = trainset.build_full_trainset()
# testset = testset.build_testset()

sim_options = {'name': 'cosine',
               'user_based': True}

algo = KNNBasic(sim_options=sim_options)
# predictions = algo.test()
# rmse(predictions)

# cross_validate(algo, trainset, cv=5, verbose=True)
# dump.dump('./dump_file', predictions, algo)

trainset, testset = train_test_split(trainset, test_size=0.25)
predictions = algo.fit(trainset,).test(testset, verbose=True)
accuracy.rmse(predictions)

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
 
best_predictions = df.sort_values(by='err')[:10]
worst_predictions = df.sort_values(by='err')[-10:]

the_predictions = df.loc[df['uid']==7000].sort_values(by='err')

print(worst_predictions)
print(the_predictions)

'''
def get_top_n(predictions, n=10):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
 
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
 
    return top_n

topNPredicted = get_top_n(predictions, n=20)
 
#打印为每个用户推荐的10部电影和对它们的评分

for uid, user_ratings in topNPredicted.items():
    if uid == 7000:
      print(uid, [(iid,round(rating,1)) for (iid, rating) in user_ratings])
'''



