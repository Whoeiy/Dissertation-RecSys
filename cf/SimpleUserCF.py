# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 23:10:41 2021

@author: JenniferYu
"""


from classes.Mpd import Mpd
import pandas as pd
import heapq
from collections import defaultdict
from operator import itemgetter
from surprise import KNNBasic


testSubject = 0
k = 10


mpd = Mpd()
'''
    1. return the user-item rating matrix
'''
data = mpd.loadMpdLatest()
# build up a rating data set
trainSet = data.build_full_trainset()


'''
    2. build up the user-user similarity matrix
'''
sim_options = {'name': 'cosine',
               'user_based': True}
model = KNNBasic(sim_options=sim_options)
model.fit(trainSet)
simsMatrix = model.compute_similarities()
# print(simsMatrix)

'''
    3. look up similar users
'''
# Get top N similar users to our test subject(0)
# (Alternate approach would be to select users up to some similarity threshold - try it!)
testUserInnerID = trainSet.to_inner_uid(testSubject)
# print(testUserInnerID)
similarityRow = simsMatrix[testUserInnerID]

# print(similarityRow)

similarUsers = []
# print(list(enumerate(similarityRow)))
for innerID, score in enumerate(similarityRow):
    if (innerID != testUserInnerID):
        similarUsers.append( (innerID, score) )

# quickly sort all of the users by their similarity to user 0
# & pluck off the top k results to get our neighborhood of similar users
kNeighbors = heapq.nlargest(k, similarUsers, key=lambda t: t[1])
# kNeighbors = heapq.nlargest(k, similarUsers, key=similarUsers.score)   --error
print("kNeighbors")
print(kNeighbors)

'''
    4. candidate generation and scoring
'''
# Get the stuff they rated, and add up ratings for each item, weighted by user similarity
candidates = defaultdict(float)
for similarUser in kNeighbors:
    innerID = similarUser[0]
    userSimilarityScore = similarUser[1]
    theirRatings = trainSet.ur[innerID] # 会返回这个用户的评分list，以(item_inner_id, rating)的格式
    for rating in theirRatings:     # rating[0]: item_inner_id, rating[1]: rating
        candidates[rating[0]] += (rating[1] / 5.0) * userSimilarityScore
print("candidates\n")
print(candidates)

'''
    5. candidate filtering
'''
# Build a dictionary of stuff the user has already seen
watched = {}
for itemID, rating in trainSet.ur[testUserInnerID]:
    watched[itemID] = 1
    
# Get top-rated items from similar users:
pos = 0
for itemID, ratingSum in sorted(candidates.items(), key=itemgetter(1), reverse=True):
    if not itemID in watched:
        tid = trainSet.to_raw_iid(itemID)
        # print(ml.getMovieName(int(movieID)), ratingSum)
        print(tid, ratingSum)
        pos += 1
        if (pos > 10):
            break




