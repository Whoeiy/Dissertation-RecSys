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
import numy as np

class Playlist:
    
    # test
    
    plsSlicePath = '../data_csv/playlists.csv'
    
    
    def loadPlayListLatest(self):
        
        # Look for files relative to the directory we are running from
        os.chdir(os.path.dirname(sys.argv[0]))
        
        