# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 21:15:41 2021

@author: JenniferYu
"""

import os
import sys
import vaex

class getData:
    pntPath_hdf5 = '../data/hdf5/playlist_tracks.hdf5'
    tracksPath_hdf5 = '../data/hdf5/real/tracks.hdf5'
    playlistsPath_hdf5 = '../data/hdf5/playlists_info.hdf5'
    
    def getPlaylists(self, retype, playlistsPath_hdf5):
        # Look for files relative to the directory we are running from
        os.chdir(os.path.dirname(sys.argv[0]))
    
        self.playlist_col = ['pid', 'name', 'collaborative', 'modified_at', 'num_tracks', 'num_albums', 'num_followers', 'num_edits', 'duration_ms', 'num_artists', 'test_type']
        
        # using vaex
        df_vaex = vaex.open(playlistsPath_hdf5)
        df_playlists = df_vaex.to_pandas_df(self.playlist_col)
        
        if retype == 1:
            exclude = [0,2]
            res = df_playlists.loc[~df_playlists['test_type'].isin(exclude), 'pid'].tolist()
            print("!!", len(res))
            return res
        if retype == 0:
            return df_playlists
    
    def getTracks(self, tracksPath_hdf5):
        # Look for files relative to the directory we are running from
        os.chdir(os.path.dirname(sys.argv[0]))
        
        # self.tid_to_tname = {}
        # self.tname_to_tid = {}
        self.track_col = ['track_uri', 'track_name', 'artist_uri', 'artist_name', 'album_uri', 'album_name', 'duration_ms', 'tid']
        
        # using vaex
        df_vaex = vaex.open(tracksPath_hdf5)
        df_tracks = df_vaex.to_pandas_df(self.track_col)
        
        return df_tracks
    
    def getResultDf(self, path):
        
        os.chdir(os.path.dirname(sys.argv[0]))
        
        df_vaex = vaex.open(path)
        df_res = df_vaex.to_pandas_df()
        
        return df_res
    
    
