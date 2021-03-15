#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 10:49:02 2021

@author: YU YUE

"""

import pandas as pd
import os
import json

# jsonfile_path = "../data_json/mpd.slice.0-999.json"
data_json_path = "../data_json"

playlist_col = ['pid', 'name', 'collaborative', 'modified_at', 'num_tracks', 'num_albums', 'num_followers', 'num_edits', 'duration_ms', 'num_artists']
tracks_col = ['track_uri', 'track_name', 'artist_uri', 'artist_name', 'album_uri', 'album_name', 'duration_ms']

# 
filenames = os.listdir(data_json_path)

#
data_playlists = []
data_tracks = []

playlists = []
tracks = set()


# for filename in filenames:
    
    # 查看data_json目录下所有的文件，过滤掉隐藏文件
for root, dirs, files in os.walk(data_json_path):
    files = [f for f in files if not f[0] == '.']
    dirs[:] = [d for d in dirs if not d[0] == '.']
    
    for filename in files:
        fullpath = os.path.join(root, filename)

    # 加载json数据文件
    with open(fullpath, encoding="utf-8") as json_obj:
        mpd_slice = json.load(json_obj)
    
    
    # 提取整理数据到data_playlists, playlists, data_tracks, tracks
    # data_playlist: 所有playlist的基本信息
    # playlists: 所有playlists中tracks的关键信息
    # data_tracks: 所有tracks的基本信息（无重复）
    # tracks: 所有tracks的track_uri信息（无重复）
    for playlist in mpd_slice['playlists']:
        data_playlists.append([playlist[col] for col in playlist_col])
        for track in playlist['tracks']:
            playlists.append([playlist['pid'], track['track_uri'], track['pos']])
            if track['track_uri'] not in tracks:
                data_tracks.append([track[col] for col in tracks_col])
                tracks.add(track['track_uri'])
                
    # 把json数据转化为dataframe格式
    # playlists_info
    df_playlists_info = pd.DataFrame(data_playlists, columns=playlist_col)
    df_playlists_info['collaborative'] = df_playlists_info['collaborative'].map({'false':False, 'true':True})
    
    # tracks
    df_tracks = pd.DataFrame(data_tracks, columns=tracks_col)
    df_tracks['tid'] = df_tracks.index
    # print(df_tracks['tid'])
    
    track_uri2tid = df_tracks.set_index('track_uri').tid
    
    # playlists
    df_playlists = pd.DataFrame(playlists, columns=['pid', 'tid', 'pos'])
    df_playlists.tid = df_playlists.tid.map(track_uri2tid)
    
    # to csv
    df_playlists_info.to_csv(r'../data_csv/test/playlists_info.csv', index=None)
    df_tracks.to_csv(r'../data_csv/test/tracks.csv', index=None)
    df_playlists.to_csv(r'../data_csv/test/playlists.csv', index=None)
    



