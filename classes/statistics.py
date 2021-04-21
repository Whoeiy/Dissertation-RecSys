# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 21:11:23 2021

@author: JenniferYu
"""
import pandas as pd
import classes.getData as getData
import os
import psutil

class Stat:
    
    def countByplength(self):
        '''
        get the number of the playlist grouped by length of tracks
        根据歌单的长度，统计其中含0、5-10、10-25、25-100、100以上首歌曲的歌单个数
        '''
        
        getdata = getData.getData()
        df_playlists = getdata.getPlaylists()
        
        
        self.df_seed_zero = df_playlists.loc[df_playlists.num_tracks == 0]
        # 0-5
        
        self.df_seed_5 = df_playlists.loc[(df_playlists.num_tracks > 0) & (df_playlists.num_tracks <= 5)]
        # 5-10
        self.df_seed_10 = df_playlists.loc[(df_playlists.num_tracks > 5) & (df_playlists.num_tracks <= 10)]
        # 10-25
        self.df_seed_25 = df_playlists.loc[(df_playlists.num_tracks > 10) & (df_playlists.num_tracks <= 25)]
        # 25-100
        self.df_seed_100 = df_playlists.loc[(df_playlists.num_tracks > 25) & (df_playlists.num_tracks <= 100)]
        # more than 100
        self.df_seed_100more = df_playlists.loc[df_playlists.num_tracks > 100]
        
        
        count = {}
        count['df_seed_zero'] = self.df_seed_zero.shape[0]
        count['df_seed_5'] = self.df_seed_5.shape[0]
        count['df_seed_10'] = self.df_seed_10.shape[0]
        count['df_seed_25'] = self.df_seed_25.shape[0]
        count['df_seed_100'] = self.df_seed_100.shape[0]
        count['df_seed_100more'] = self.df_seed_100more.shape[0]
        return count
    
    def show_ram(self):
        pid = os.getpid()
        p = psutil.Process(pid)         # 获取当前进程的pid
        info = p.memory_full_info()     # 根据pid找到该进程，进而占到占用的内存值
        memory = info.uss / 1024 / 1024
        return memory
        
        
        
        
        
        
        
        