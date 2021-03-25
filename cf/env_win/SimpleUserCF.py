# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 23:10:41 2021

@author: JenniferYu
"""


from Mpd import Mpd
import pandas as pd
import heapq
from collections import defaultdict
from operator import itemgetter
from surprise import KNNBasic


testSubject = '0'
k = 10


mpd = Mpd()
data = mpd.loadMpdLatest()

trainSet = data.build_full_trainset()

sim_options = {'name': 'cosine',
               'user_based': True}

model = KNNBasic(sim_options=sim_options)
model.fit(trainSet)
simsMatrix = model.compute_similarities()

print(simsMatrix)




