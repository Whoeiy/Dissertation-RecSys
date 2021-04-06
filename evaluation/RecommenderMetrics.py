# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 22:08:17 2021

@author: JenniferYu
"""

import itertools

from surprise import accuracy
from collections import defaultdict

class RecommenderMetrics:

    def MAE(predictions):
        return accuracy.mae(predictions, verbose=False)

    def RMSE(predictions):
        return accuracy.rmse(predictions, verbose=False)

    '''
    function GetTopN:
        predictions:
        n: 函数返回的topN中的N
        minimumRating: a threshold, 若estimatedRating>minimumRating，则该条prediction可以加入到TopN中
    '''
    def GetTopN(predictions, n, minimumRating=0):
        topN = defaultdict(list)


        for playlistID, trackID, actualRating, estimatedRating, _ in predictions:
            if (estimatedRating >= minimumRating):
                topN[int(playlistID)].append((int(trackID), estimatedRating))

        for playlistID, ratings in topN.items():
            ratings.sort(key=lambda x: x[1], reverse=True)
            topN[int(playlistID)] = ratings[:n]

        return topN

    def HitRate(topNPredicted, leftOutPredictions):
        hits = 0
        total = 0

        # For each left-out rating
        for leftOut in leftOutPredictions:
            playlistID = leftOut[0]
            leftOutTrackID = leftOut[1]
            # Is it in the predicted top 10 for this user?
            hit = False
            for trackID, predictedRating in topNPredicted[int(playlistID)]:
                if (int(leftOutTrackID) == int(trackID)):
                    hit = True
                    break
            if (hit) :
                hits += 1

            total += 1

        # Compute overall precision
        return hits/total

    def CumulativeHitRate(topNPredicted, leftOutPredictions, ratingCutoff=0):
        hits = 0
        total = 0

        # For each left-out rating
        for playlistID, leftOutTrackID, actualRating, estimatedRating, _ in leftOutPredictions:
            # Only look at ability to recommend things the users actually liked...
            if (actualRating >= ratingCutoff):
                # Is it in the predicted top 10 for this user?
                hit = False
                for trackID, predictedRating in topNPredicted[int(playlistID)]:
                    if (int(leftOutTrackID) == trackID):
                        hit = True
                        break
                if (hit) :
                    hits += 1

                total += 1

        # Compute overall precision
        return hits/total

    def RatingHitRate(topNPredicted, leftOutPredictions):
        hits = defaultdict(float)
        total = defaultdict(float)

        # For each left-out rating
        for playlistID, leftOutTrackID, actualRating, estimatedRating, _ in leftOutPredictions:
            # Is it in the predicted top N for this user?
            hit = False
            for trackID, predictedRating in topNPredicted[int(playlistID)]:
                if (int(leftOutTrackID) == trackID):
                    hit = True
                    break
            if (hit) :
                hits[actualRating] += 1

            total[actualRating] += 1

        # Compute overall precision
        for rating in sorted(hits.keys()):
            print (rating, hits[rating] / total[rating])

    def AverageReciprocalHitRank(topNPredicted, leftOutPredictions):
        summation = 0
        total = 0
        # For each left-out rating
        for playlistID, leftOutTrackID, actualRating, estimatedRating, _ in leftOutPredictions:
            # Is it in the predicted top N for this user?
            hitRank = 0
            rank = 0
            for trackID, predictedRating in topNPredicted[int(playlistID)]:
                rank = rank + 1
                if (int(leftOutTrackID) == trackID):
                    hitRank = rank
                    break
            if (hitRank > 0) :
                summation += 1.0 / hitRank

            total += 1

        return summation / total

    # What percentage of playlists have at least one "good" recommendation
    def PlaylistCoverage(topNPredicted, numPlaylists, ratingThreshold=0):
        hits = 0
        for playlistID in topNPredicted.keys():
            hit = False
            for trackID, predictedRating in topNPredicted[playlistID]:
                if (predictedRating >= ratingThreshold):
                    hit = True
                    break
            if (hit):
                hits += 1

        return hits / numPlaylists

    def Diversity(topNPredicted, simsAlgo):
        n = 0
        total = 0
        simsMatrix = simsAlgo.compute_similarities()
        for playlistID in topNPredicted.keys():
            pairs = itertools.combinations(topNPredicted[playlistID], 2)
            for pair in pairs:
                track1 = pair[0][0]
                track2 = pair[1][0]
                innerID1 = simsAlgo.trainset.to_inner_iid(str(track1))
                innerID2 = simsAlgo.trainset.to_inner_iid(str(track2))
                similarity = simsMatrix[innerID1][innerID2]
                total += similarity
                n += 1

        S = total / n
        return (1-S)

    def Novelty(topNPredicted, rankings):
        n = 0
        total = 0
        for playlistID in topNPredicted.keys():
            for rating in topNPredicted[playlistID]:
                trackID = rating[0]
                rank = rankings[trackID]
                total += rank
                n += 1
        return total / n
