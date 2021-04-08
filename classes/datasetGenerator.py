#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 14:44:51 2021

@author: whoeiy
"""
import os
import json
import math
import pandas as pd

class small_trainset:
    
    '''
    generate a small train data set(just for debug)
    '''
    def __init__(self, data_json_path, output_type):
    # def __init__(self, data_json_path, save_type, save_dir, last_slice_tid):
        
        self.playlist_col = ['pid', 'name', 'collaborative', 'modified_at', 'num_tracks', 'num_albums', 'num_followers', 'num_edits', 'duration_ms', 'num_artists']
        self.tracks_col = ['track_uri', 'track_name', 'artist_uri', 'artist_name', 'album_uri', 'album_name', 'duration_ms']
        
        self.data_playlists = list()
        self.data_tracks = list()
        
        self.playlist_tracks = []
        self.tracks = set()
        
        self.tracklen_pp = []
        self.test_playlist_tracks = []
        
        self.output_type = output_type
        
        # self.last_slice_tid = last_slice_tid
        # self.this_slice_tid = int()
        
            # 查看data_json目录下所有的文件，过滤掉隐藏文件
        for root, dirs, files in os.walk(data_json_path):
            files = [f for f in files if not f[0] == '.']
            dirs[:] = [d for d in dirs if not d[0] == '.']
            
            for filename in files:
                fullpath = os.path.join(root, filename)
                # 加载json数据文件
                with open(fullpath, encoding="utf-8") as json_obj:
                    mpd_slice = json.load(json_obj)
                    print("processing file: "+str(filename))
                    self.generator(mpd_slice)
        if output_type == 1:
            # csv
            print("writing into the csv file...")
            self.jsonToCSV()
        else:
            # hdf5
            print("writing into the hdf5 file...")
            self.jsonToHDF5()
        print("done.")
        

            
            
    
    def generator(self, mpd_slice):
        # 提取整理数据到data_playlists, playlist_tracks, data_tracks, tracks
        # data_playlist: 所有playlist的基本信息
        # playlist_tracks: 所有playlists中tracks的关键信息
        # data_tracks: 所有tracks的基本信息（无重复）
        # tracks: 所有tracks的track_uri信息（无重复）
        
        for playlist in mpd_slice['playlists']:
            self.data_playlists.append([playlist[col] for col in self.playlist_col])
            # self.tracklen_pp.append([playlist['pid'],len(playlist['tracks']), math.floor(len(playlist['tracks'])*0.8)])
            for track in playlist['tracks']:
                self.playlist_tracks.append([playlist['pid'], track['track_uri'], '1', track['pos']])
                if track['track_uri'] not in self.tracks:
                    self.data_tracks.append([track[col] for col in self.tracks_col])
                    self.tracks.add(track['track_uri'])
                
        ''' 手动划分testset - needless
        for playlist in mpd_slice['playlists']:
            self.data_playlists.append([playlist[col] for col in self.playlist_col])
            self.tracklen_pp.append([playlist['pid'],len(playlist['tracks']), math.floor(len(playlist['tracks'])*0.8)])
            for track in playlist['tracks']:
                if track['pos'] > math.floor(len(playlist['tracks'])*0.8):
                    # belong to test dataset
                    print('belong to test dataset')
                    self.test_playlist_tracks.append([playlist['pid'], track['track_uri'], '1', track['pos']])
                else:
                    # belong to train dataset
                    print('belong to train dataset')
                    self.playlist_tracks.append([playlist['pid'], track['track_uri'], '1', track['pos']])
                if track['track_uri'] not in self.tracks:
                    self.data_tracks.append([track[col] for col in self.tracks_col])
                    self.tracks.add(track['track_uri'])
        '''
                    
                    
    def jsonToCSV(self):
        
        # playlists_info
        df_playlists_info = pd.DataFrame(self.data_playlists, columns=self.playlist_col)
        df_playlists_info['collaborative'] = df_playlists_info['collaborative'].map({'false':False, 'true':True})
        
        # tracks
        df_tracks = pd.DataFrame(self.data_tracks, columns=self.tracks_col)
        # df_tracks['tid'] = self.last_slice_tid + df_tracks.index
        df_tracks['tid'] = df_tracks.index
        # print(df_tracks['tid'])
        
        track_uri2tid = df_tracks.set_index('track_uri').tid
        
        # df_tracklen_pp = pd.DataFrame(self.tracklen_pp, columns=['pid', 'raw_tracks_len', 'train_tracks_len'])
        # print(df_tracklen_pp.head(10))
        
        # playlist_tracks
        df_playlist_tracks = pd.DataFrame(self.playlist_tracks, columns=['pid', 'tid', 'rating', 'pos'])
        # df_playlist_tracks = pd.DataFrame(playlist_tracks, columns=['user', 'item', 'rating', 'pos'])
        df_playlist_tracks.tid = df_playlist_tracks.tid.map(track_uri2tid)
        # df_playlist_tracks.item = df_playlist_tracks.item.map(track_uri2tid)        
        
        df_playlist_tracks_count = df_playlist_tracks.groupby(['pid', 'tid'], as_index=False)['rating'].count()
        
        # playlist_tracks for test  needless
        # df_test_playlist_tracks = pd.DataFrame(self.test_playlist_tracks, columns=['pid', 'tid', 'rating', 'pos'])
        # df_test_playlist_tracks.tid = df_test_playlist_tracks.tid.map(track_uri2tid)
        '''
        for pid in df_playlist_tracks_count['pid']:
            df_temp = df_tracklen_pp.loc[df_tracklen_pp['pid']==pid]
            # print(df_temp)
            n = int(df_temp['raw_tracks_len']) - int(df_temp['train_tracks_len'])
            # print(n)
            df_test_playlist_tracks.append(df_playlist_tracks.groupby('pid').tail(n))
        print("test_dataset_gneration: done.")
        '''
        # to csv
        df_playlists_info.to_csv(r'../data_csv/test/playlists_info.csv', index=None)
        df_tracks.to_csv(r'../data_csv/test/tracks.csv', index=None)
        df_playlist_tracks.to_csv(r'../data_csv/test/playlist_tracks_raw.csv', index=None)
        df_playlist_tracks_count.to_csv(r'../data_csv/test/playlist_tracks.csv')
        # df_test_playlist_tracks.to_csv(r'../data_csv/test/test_playlist_tracks.csv')
        
    def jsonToHDF5(self):
        # playlists_info
        df_playlists_info = pd.DataFrame(self.data_playlists, columns=self.playlist_col)
        df_playlists_info['collaborative'] = df_playlists_info['collaborative'].map({'false':False, 'true':True})
        
        # tracks
        df_tracks = pd.DataFrame(self.data_tracks, columns=self.tracks_col)
        # df_tracks['tid'] = self.last_slice_tid + df_tracks.index
        df_tracks['tid'] = df_tracks.index
        # print(df_tracks['tid'])
        
        track_uri2tid = df_tracks.set_index('track_uri').tid
        
        # df_tracklen_pp = pd.DataFrame(self.tracklen_pp, columns=['pid', 'raw_tracks_len', 'train_tracks_len'])
        # print(df_tracklen_pp.head(10))
        
        # playlist_tracks
        df_playlist_tracks = pd.DataFrame(self.playlist_tracks, columns=['pid', 'tid', 'rating', 'pos'])
        # df_playlist_tracks = pd.DataFrame(playlist_tracks, columns=['user', 'item', 'rating', 'pos'])
        df_playlist_tracks.tid = df_playlist_tracks.tid.map(track_uri2tid)
        # df_playlist_tracks.item = df_playlist_tracks.item.map(track_uri2tid)        
        
        df_playlist_tracks_count = df_playlist_tracks.groupby(['pid', 'tid'], as_index=False)['rating'].count()
        
        hdf5_path = "../data_hdf5"
        store = pd.HDFStore("../data_hdf5/playlists_info.hdf5")
        store.put(key='playlists_info', value=df_playlists_info)
        store.close()
        
        store = pd.HDFStore("../data_hdf5/tracks.hdf5")
        store.put(key='tracks', value=df_tracks)
        store.close()
        
        store = pd.HDFStore("../data_hdf5/playlist_tracks.hdf5")
        store.put(key='playlist_tracks', value=df_playlist_tracks_count)
        store.close()


# sclass test_dataset:
    
