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
import random
import vaex

class extractor:
    
    '''
    generate a small train data set(just for debug)
    '''
    def __init__(self, data_json_path, dataset_type, output_type):
    # def __init__(self, data_json_path, save_type, save_dir, last_slice_tid):
        
        self.playlist_col = ['pid', 'name', 'collaborative', 'modified_at', 'num_tracks', 'num_albums', 'num_followers', 'num_edits', 'duration_ms', 'num_artists']
        self.track_col = ['track_uri', 'track_name', 'artist_uri', 'artist_name', 'album_uri', 'album_name', 'duration_ms']
        
        # train
        self.data_playlists = list()
        self.data_tracks = list()
        
        self.playlist_tracks = []
        self.tracks = set()
        
        self.tracklen_pp = []
        # testset
        self.test_playlist_tracks = []
        self.test_tracks = []
        
        self.dataset_type = {1:'raw dataset', 2:'testset & train set', 3:'challenge set'}
        self.output_type = output_type
        
        # self.last_slice_tid = last_slice_tid
        # self.this_slice_tid = int()
        
        print("type of the generating dataset: " + self.dataset_type[dataset_type])
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
                    self.rawdata(mpd_slice)

        
        if dataset_type == 1:
            self.jsonToDf()
        elif dataset_type == 2:
            self.jsonToDf()
            self.testset()
            
        if output_type == 1:
            # csv
            print("writing into the csv file...")
            self.jsonToCSV()
        else:
            # hdf5
            print("writing into the hdf5 file...")
            self.jsonToHDF5()
        print("done.")
        

            
            
    
    def rawdata(self, mpd_slice):
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
                    self.data_tracks.append([track[col] for col in self.track_col])
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
    def jsonToDf(self):
        # json to dataframe
        # playlists_info
        self.df_playlists_info = pd.DataFrame(self.data_playlists, columns=self.playlist_col)
        self.df_playlists_info['collaborative'] = self.df_playlists_info['collaborative'].map({'false':False, 'true':True})
        
        # tracks
        self.df_tracks = pd.DataFrame(self.data_tracks, columns=self.track_col)
        # df_tracks['tid'] = self.last_slice_tid + df_tracks.index
        self.df_tracks['tid'] = self.df_tracks.index
        # print(df_tracks['tid'])
        
        track_uri2tid = self.df_tracks.set_index('track_uri').tid
        
        # df_tracklen_pp = pd.DataFrame(self.tracklen_pp, columns=['pid', 'raw_tracks_len', 'train_tracks_len'])
        # print(df_tracklen_pp.head(10))
        
        # playlist_tracks
        self.df_playlist_tracks = pd.DataFrame(self.playlist_tracks, columns=['pid', 'tid', 'rating', 'pos'])
        # df_playlist_tracks = pd.DataFrame(playlist_tracks, columns=['user', 'item', 'rating', 'pos'])
        self.df_playlist_tracks.tid = self.df_playlist_tracks.tid.map(track_uri2tid)
        # df_playlist_tracks.item = df_playlist_tracks.item.map(track_uri2tid)        
        
        
        self.df_playlist_tracks_count = self.df_playlist_tracks.groupby(['pid', 'tid'], as_index=False)['rating'].count() 
    
    def testset(self):
        
        # testset size: 8K
        
        # >100
        df_seed_100more = self.df_playlists_info.loc[self.df_playlists_info.num_tracks > 100]
        # 随机选择1000个包含100首以上歌曲的playlist
        print(df_seed_100more.shape[0])
        df_test_p = df_seed_100more.sample(n=2000)
        pid2pnt = df_test_p['pid'].tolist()
        self.test_pid = pid2pnt
        print(len(pid2pnt))
        # df_test_pnt = self.df_playlist_tracks_count.loc[self.df_playlist_tracks_count['pid'].isin(pid2pnt)]
        df_show = self.df_playlist_tracks_count.loc[self.df_playlist_tracks_count['pid'].isin(pid2pnt)].groupby('pid').head(100)
        # self.df_test_pnt = self.df_playlist_tracks_count.loc[self.df_playlist_tracks_count['pid'].isin(pid2pnt)].groupby('pid').head(100)
        # print(self.df_test_pnt.head(50))
        print('only have 100:')
        print(df_show.shape[0])
        # 风险点：会改变track在歌单中的顺序，但是还有pos字段
        
        
        
        # 0-5
        df_seed_5less = self.df_playlists_info.loc[self.df_playlists_info.num_tracks <= 5]
        # 随机选择1000个包含100首以上歌曲的playlist
        print(df_seed_100more.shape[0])
        pid2pnt += df_seed_5less['pid'].tolist()
        print("notice:", len(pid2pnt))
        # df_test_pnt = self.df_playlist_tracks_count.loc[self.df_playlist_tracks_count['pid'].isin(pid2pnt)]
        df_pnt_candidate = self.df_playlist_tracks_count.loc[~self.df_playlist_tracks_count['pid'].isin(pid2pnt)]
        # print(df_test_pnt.shape[0])
        df_pnt_candidate = df_pnt_candidate.sample(n=6000)
        # 5
        pid2pnt = df_pnt_candidate[0:2000]['pid'].tolist()
        df_pnt_candi_selected = self.df_playlist_tracks_count.loc[self.df_playlist_tracks_count['pid'].isin(pid2pnt)].groupby('pid').head(5)
        # df_show.append(df_pnt_candi_selected)
        print('all5:', df_pnt_candi_selected.shape[0])
        # 10
        pid2pnt = df_pnt_candidate[2000:4000]['pid'].tolist()
        df_pnt_candi_selected = self.df_playlist_tracks_count.loc[self.df_playlist_tracks_count['pid'].isin(pid2pnt)].groupby('pid').head(10)
        print('all10:', df_pnt_candi_selected.shape[0])
        # 25
        pid2pnt = df_pnt_candidate[4000:]['pid'].tolist()
        df_pnt_candi_selected = self.df_playlist_tracks_count.loc[self.df_playlist_tracks_count['pid'].isin(pid2pnt)].groupby('pid').head(25)
        print('all25:', df_pnt_candi_selected.shape[0])
        
        
        
                    
    def jsonToCSV(self):
        
        # playlists_info
        df_playlists_info = pd.DataFrame(self.data_playlists, columns=self.playlist_col)
        df_playlists_info['collaborative'] = df_playlists_info['collaborative'].map({'false':False, 'true':True})
        
        # tracks
        df_tracks = pd.DataFrame(self.data_tracks, columns=self.track_col)
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
        
    def jsonToHDF5(self, dataset_type):
        
        if dataset_type == 1:
            hdf5_path = "../data/hdf5_0419/"
            # plylists_info
            vaex_df = vaex.from_pandas(self.df_playlists_info, copy_index=False)
            vaex_df.export_hdf5(hdf5_path+'playlists_info.hdf5')
            # tracks
            vaex_df = vaex.from_pandas(self.df_tracks, copy_index=False)
            vaex_df.export_hdf5(hdf5_path+'tracks.hdf5') 
            # playlist_tracks
            vaex_df = vaex.from_pandas(self.df_playlist_tracks_count, copy_index=False)
            vaex_df.export_hdf5(hdf5_path+'playlist_tracks.hdf5')
        elif dataset_type == 2:
            hdf5_test_path = '../data/testset/0419/'
            
            # testset
            # vaex_df = vaex.from_pandas(self.)
                


        

        
    




    
