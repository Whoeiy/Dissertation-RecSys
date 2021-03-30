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

from collections import defaultdict
import numpy as np
import pandas as pd

class Mpd:
    
    # test - slice file
    pntPath = '../data_csv/test/playlist_tracks.csv'
    trackPath = '../data_csv/test/tracks.csv'
    playlistPath = '../data_csv/test/playlists_info.csv'
    
    
    def loadMpdLatest(self):
        
        # Look for files relative to the directory we are running from
        os.chdir(os.path.dirname(sys.argv[0]))
        
        self.pid_to_pname = {}
        self.pname_to_pid = {}
        
        # df = pd.read_csv(self.pntPath, usecols=['pid', 'tid', 'rating'])
        df = pd.read_csv(self.pntPath, usecols=['pid', 'tid', 'rating'])
        df.columns = ['user', 'item', 'rating']
        print(df.head(10))
        reader = Reader(rating_scale=(0,1))
        ratingsDataset = Dataset.load_from_df(df[['user', 'item', 'rating']], reader)
        
        
        return ratingsDataset
    
        
        """
        
        
        
        ratingsDataset = Dataset.load_from_file(self.pntPath, reader=reader)
        with open(self.playlistPath, newline='', encoding='ISO-8859-1') as csvfile:
            playlistReader = csv.reader(csvfile)
            next(playlistReader) # skip header line
            for row in playlistReader:
                pid = row[0]
                tid = row[1]
                
        
    
    def playlistRatings(self, pid):
        playlistRatings = []
        
        with open(self.)
        """
        