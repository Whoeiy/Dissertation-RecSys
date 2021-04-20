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
    tracksPath_hdf5 = '../data/hdf5_500K/tracks.hdf5'
    playlistsPath_hdf5 = '../data/hdf5/playlists_info.hdf5'
    
    def getPlaylists(self):
        # Look for files relative to the directory we are running from
        os.chdir(os.path.dirname(sys.argv[0]))
    
        self.playlist_col = ['pid', 'name', 'collaborative', 'modified_at', 'num_tracks', 'num_albums', 'num_followers', 'num_edits', 'duration_ms', 'num_artists']
        
        # using vaex
        df_vaex = vaex.open(self.playlistsPath_hdf5)
        df_playlists = df_vaex.to_pandas_df(self.playlist_col)
        
        return df_playlists
    
    def getTracks(self):
        # Look for files relative to the directory we are running from
        os.chdir(os.path.dirname(sys.argv[0]))
        
        # self.tid_to_tname = {}
        # self.tname_to_tid = {}
        self.track_col = ['track_uri', 'track_name', 'artist_uri', 'artist_name', 'album_uri', 'album_name', 'duration_ms', 'tid']
        
        # using vaex
        df_vaex = vaex.open(self.tracksPath_hdf5)
        df_tracks = df_vaex.to_pandas_df(self.track_col)
        
        return df_tracks
    
    
