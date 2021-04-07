#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 13:54:08 2021

@author: YUE YU
"""

import os
import csv
import sys
import re

from surprise import Dataset
from surprise import Reader
from surprise import BaselineOnly
from surprise import NormalPredictor
from surprise.model_selection import cross_validate

from collections import defaultdict
import numpy as np
import pandas as pd

class Mpd:
    
    # test - slice file
    pntPath = '../data_csv/test/playlist_tracks.csv'
    test_pntPath = '../data_csv/test/test_playlist_tracks.csv'
    trackPath = '../data_csv/test/tracks.csv'
    playlistPath = '../data_csv/test/playlists_info.csv'
    
    def getPntDf(self):
        os.chdir(os.path.dirname(sys.argv[0]))
        
        df_pnt = pd.read_csv(self.pntPath, usecols=['pid', 'tid', 'rating'])
        df_pnt.columns = ['user', 'item', 'rating']
        
        df_pnt.head(20)
        
        return df_pnt
        
    def loadMpdLatest(self):
        
        # Look for files relative to the directory we are running from
        os.chdir(os.path.dirname(sys.argv[0]))
        
        self.tid_to_tname = {}
        self.tname_to_tid = {}
        
        # df = pd.read_csv(self.pntPath, usecols=['pid', 'tid', 'rating'])
        df_pnt = pd.read_csv(self.pntPath, usecols=['pid', 'tid', 'rating'])
        df_pnt.columns = ['user', 'item', 'rating']
        # print(df.head(10))
        reader = Reader(rating_scale=(0,1))
        ratingsDataset = Dataset.load_from_df(df_pnt[['user', 'item', 'rating']], reader)
        
        cross_validate(NormalPredictor(), ratingsDataset, cv=2, verbose=True)
        # cross_validate(BaselineOnly(), ratingsDataset, verbose=True)
        return ratingsDataset
        '''
            have problem -- start
        
        df_t = pd.read_csv(self.trackPath)
        self.tid_to_tname = df_t.groupby('tid')['track_name'].apply(lambda x:str(x)).to_dict()
        print(self.tid_to_tname.values())
          
        
    
    def getTrackName(self, tid):
        return self.tid_to_tname.get(tid, "track name not found")
        
            end
        '''
    def loadMpdTrainset(self):
        # Look for files relative to the directory we are running from
        os.chdir(os.path.dirname(sys.argv[0]))
        
        self.tid_to_tname = {}
        self.tname_to_tid = {}
        
        # df = pd.read_csv(self.pntPath, usecols=['pid', 'tid', 'rating'])
        df_pnt = pd.read_csv(self.pntPath, usecols=['pid', 'tid', 'rating'])
        df_pnt.columns = ['user', 'item', 'rating']
        # print(df.head(10))
        reader = Reader(rating_scale=(0,1))
        trainDataset = Dataset.load_from_df(df_pnt[['user', 'item', 'rating']], reader)
        
        # cross_validate(BaselineOnly(), ratingsDataset, verbose=True)
        return trainDataset
    
    def loadMpdDataDf(self):
        # Look for files relative to the directory we are running from
        os.chdir(os.path.dirname(sys.argv[0]))
        
        self.tid_to_tname = {}
        self.tname_to_tid = {}
        
        # df = pd.read_csv(self.pntPath, usecols=['pid', 'tid', 'rating'])
        df_pnt = pd.read_csv(self.pntPath, usecols=['pid', 'tid', 'rating'])
        df_pnt.columns = ['user', 'item', 'rating']
        # print(df.head(10))
        # reader = Reader(rating_scale=(0,1))
        # trainDataset = Dataset.load_from_df(df_pnt[['user', 'item', 'rating']], reader)
        
        # cross_validate(BaselineOnly(), ratingsDataset, verbose=True)
        return df_pnt
    
    
    def getPopularityRanks(self):
        ratings = defaultdict(int)
        rankings = defaultdict(int)
        
        df_pnt = pd.read_csv(self.pntPath, usecols=['pid', 'tid', 'rating'])
        df_pnt.columns = ['user', 'item', 'rating']
        
        with open(self.ratingsPath, newline='') as csvfile:
            ratingReader = csv.reader(csvfile)
            next(ratingReader)
            for row in ratingReader:
                trackID = int(row[1])
                ratings[trackID] += 1
        rank = 1
        for movieID, ratingCount in sorted(ratings.items(), key=lambda x: x[1], reverse=True):
            rankings[movieID] = rank
            rank += 1
        return rankings
    
       
        
        
        