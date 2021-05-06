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
import time

class extractor:
    
    '''
    extractor类：生成训练用数据集
    '''
    def __init__(self, data_json_path, dataset_type, output_type):
    # def __init__(self, data_json_path, save_type, save_dir, last_slice_tid):
        
        print('\n=======')
        print('生成数据集(extractor类)')
        print('=======\n')
        
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
        
        print("- 生成数据集的形式: " + self.dataset_type[dataset_type])
        # 查看data_json目录下所有的文件，过滤掉隐藏文件
        for root, dirs, files in os.walk(data_json_path):
            files = [f for f in files if not f[0] == '.']
            dirs[:] = [d for d in dirs if not d[0] == '.']
            
            for filename in files:
                fullpath = os.path.join(root, filename)
                # 加载json数据文件
                with open(fullpath, encoding="utf-8") as json_obj:
                    mpd_slice = json.load(json_obj)
                    print("- 读取文件: "+str(filename))
                    self.rawdata(mpd_slice)

        
        if dataset_type == 1:
            self.jsonToDf()
        elif dataset_type == 2:
            self.jsonToDf()
            self.test_train_set()
            
        if output_type == 1:
            # csv
            print("- 写入csv文件...")
            self.jsonToCSV(dataset_type)
        else:
            # hdf5
            print("- 写入hdf5文件...")
            self.jsonToHDF5(dataset_type)
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
                    
                
    
    def jsonToDf(self):
        # json to dataframe
        # playlists_info
        self.df_playlists_info = pd.DataFrame(self.data_playlists, columns=self.playlist_col)
        self.df_playlists_info['collaborative'] = self.df_playlists_info['collaborative'].map({'false':False, 'true':True})
        self.df_playlists_info_copy = self.df_playlists_info.copy()
        # tracks
        self.df_tracks = pd.DataFrame(self.data_tracks, columns=self.track_col)
        # df_tracks['tid'] = self.last_slice_tid + df_tracks.index
        self.df_tracks['tid'] = self.df_tracks.index
        # print(self.df_tracks['tid'])
        
        track_uri2tid = self.df_tracks.set_index('track_uri').tid
        
        # df_tracklen_pp = pd.DataFrame(self.tracklen_pp, columns=['pid', 'raw_tracks_len', 'train_tracks_len'])
        # print(df_tracklen_pp.head(10))
        
        # playlist_tracks
        self.df_playlist_tracks = pd.DataFrame(self.playlist_tracks, columns=['pid', 'tid', 'rating', 'pos'])
        # df_playlist_tracks = pd.DataFrame(playlist_tracks, columns=['user', 'item', 'rating', 'pos'])
        self.df_playlist_tracks.tid = self.df_playlist_tracks.tid.map(track_uri2tid)
        # df_playlist_tracks.item cccc= df_playlist_tracks.item.map(track_uri2tid)        
        
        
        self.df_playlist_tracks_count = self.df_playlist_tracks.groupby(['pid', 'tid'], as_index=False)['rating'].count() 
    
    def test_train_set(self):
        
        # testset size: 10K
        print("****** 生成测试集 ******")
        
        # >25
        print("** seed>25(500 playlists) **")
        df_seed_25more = self.df_playlists_info.loc[self.df_playlists_info.num_tracks > 25]
        # 随机选择1000个包含100首以上歌曲的playlist
        print("* 播放列表数（数据集中含25+首歌曲）: ", df_seed_25more.shape[0])
        df_test_p = df_seed_25more.sample(n=5000, replace=False, random_state=1)
        pid2pnt = list(set(df_test_p['pid'].tolist()))
        test_pid = pid2pnt
        
        # print("* 播放列表数（测试集中含25首歌曲）: ", len(pid2pnt))
        # print("* 播放列表数（测试集）: ", len(test_pid))  
        print("\n* 随机选择5000个含25+首歌曲的播放列表")
        print("* 分别生成1000个含1首歌曲、1000个含5首歌曲、1000个含10首歌曲、2000个含25首歌曲的播放列表\n")
        print("* 选择前n首歌曲，播放列表剩余的歌曲作为测试集的实际结果\n")
        
        pid2pnt_group = {1:pid2pnt[0:1000], 5:pid2pnt[1000:2000], 10:pid2pnt[2000:3000], 25:pid2pnt[3000:]}
        
        
        self.df_testset = pd.DataFrame(columns=['A', 'B', 'C', 'D'])
        self.df_trueset = pd.DataFrame(columns=['A', 'B', 'C', 'D'])
        for i in pid2pnt_group.keys(): 
            self.df_playlists_info_copy.loc[self.df_playlists_info_copy['pid'].isin(pid2pnt_group[i]), 'test_type'] = i
            df_help = self.df_playlist_tracks.loc[self.df_playlist_tracks['pid'].isin(pid2pnt_group[i])].copy()
            all_index = df_help.index.tolist()
            df_chosen = df_help.groupby('pid').head(i)
            chosen_index = df_help.groupby('pid').head(i).index.tolist()
            print(len(chosen_index))
            # chosen_tid = df_chosen['tid'].tolist()
            
            true_index = list(set(all_index) - set(chosen_index))
            df_true = df_help.loc[[k for k in true_index]]
            # df_true = df_help.loc[(~df_help[index].isin(chosen_pid) and df_help['tid'].isin(chosen_tid))]
        
            # df_true = df_help.loc[~df_help['tid'].isin(chosen_tid)]
            print("* 播放列表数（测试集中含", i, "首歌曲）: ", df_chosen.shape[0])
            # print("* only have ", i, ": ", df_chosen.shape[0])
            if i == 1:
                self.df_testset = pd.DataFrame(df_chosen)
                self.df_trueset = pd.DataFrame(df_true)
            else:
                self.df_testset = self.df_testset.append(df_chosen)
                self.df_trueset = self.df_trueset.append(df_true)
        # print("* testset length: ", self.df_testset.shape[0])
        print("* 播放列表数（测试集）: ", self.df_testset.shape[0]) 

        
        
        # >100
        print("** seed>100(2000 playlists) **")
        df_seed_100more = self.df_playlists_info.loc[~self.df_playlists_info['pid'].isin(test_pid)].loc[self.df_playlists_info.num_tracks > 100]
        # df_seed_100more = self.df_playlists_info.loc[self.df_playlists_info.num_tracks > 100]
        
        # 随机选择1000个包含100首以上歌曲的playlist
        print("* 播放列表数（数据集中含100+首歌曲）: ", df_seed_100more.shape[0])
        df_test_p = df_seed_100more.sample(n=2000, replace=False, random_state=1)
        pid2pnt = list(set(df_test_p['pid'].tolist()))
        test_pid.extend(pid2pnt)
               
        # print("* chosen 100-playlists for testset: ", len(pid2pnt))
        # print("* chosen playlists for testset: ", len(test_pid))
        print("\n* 随机选择2000个含100+首歌曲的播放列表")
        print("* 生成2000个含100首歌曲的播放列表\n")
        self.df_playlists_info_copy.loc[self.df_playlists_info_copy['pid'].isin(pid2pnt), 'test_type'] = 100
        df_help = self.df_playlist_tracks.loc[self.df_playlist_tracks['pid'].isin(pid2pnt)].copy()
        all_index = df_help.index.tolist()
        df_chosen = df_help.groupby('pid').head(100)
        chosen_index = df_help.groupby('pid').head(100).index.tolist()
        print(len(chosen_index))
        # chosen_tid = df_chosen['tid'].tolist()
        
        true_index = list(set(all_index) - set(chosen_index))
        df_true = df_help.loc[[k for k in true_index]]        
        # df_true = df_help.loc[~df_help['tid'].isin(chosen_tid)]
        # print('100: ', df_true.head())
        print('* 播放列表数（测试集中含100首歌曲）:', df_chosen.shape[0])
        self.df_testset = self.df_testset.append(df_chosen)
        self.df_trueset = self.df_trueset.append(df_true)
        print("* 播放列表数（测试集）: ", self.df_testset.shape[0])
        
        
        # random
        print("** no tittle(2000 playlists) & 0 seed(1000 playlists) **")
        df_rest = self.df_playlists_info.loc[~self.df_playlists_info['pid'].isin(test_pid)].loc[self.df_playlists_info.num_tracks > 0]
        # 随机选择1000个包含100首以上歌曲的playlist
        print("* 播放列表数（数据集中剩余的）: ", df_rest.shape[0])
        df_test_p = df_rest.sample(n=3000, replace=False, random_state=1)
        pid2pnt = list(set(df_test_p['pid'].tolist()))
        test_pid.extend(pid2pnt)
         
        # print("* chosen rest-playlists for testset: ", len(pid2pnt))
        # print("* chosen playlists for testset: ", len(test_pid)) 
        print("\n* 随机选择3000个剩余的播放列表")
        print("* 分别生成1000个含0首歌曲、2000个没有标题的播放列表\n")
        
        self.test_pid = test_pid
        
        pid2pnt_group = {'zero':pid2pnt[0:1000], 'nt':pid2pnt[1000:]}
        
        # zero seed
        self.df_playlists_info_copy.loc[self.df_playlists_info_copy['pid'].isin(pid2pnt_group['zero']), 'test_type'] = 0
        self.zero_seed_pid = pid2pnt_group['zero']
        df_true = self.df_playlist_tracks.loc[self.df_playlist_tracks['pid'].isin(self.zero_seed_pid)].copy()
        print('播放列表数（测试集中不含歌曲的）: ', len(self.zero_seed_pid))
        self.df_trueset = self.df_trueset.append(df_true)
        
        # no tittle
        self.df_playlists_info_copy.loc[self.df_playlists_info_copy['pid'].isin(pid2pnt_group['nt']), 'name'] = ''
        self.df_playlists_info_copy.loc[self.df_playlists_info_copy['pid'].isin(pid2pnt_group['nt']), 'test_type'] = 2
        df_chosen = self.df_playlist_tracks.loc[self.df_playlist_tracks['pid'].isin(pid2pnt_group['nt'])]
        print("* 播放列表数（测试集中没有标题的）: ", df_chosen.shape[0])
        self.df_testset = self.df_testset.append(df_chosen)
        print("* 播放列表数（测试集）: ", self.df_testset.shape[0])
        
        # for output
        self.df_playlists_info_test = self.df_playlists_info_copy.loc[self.df_playlists_info['pid'].isin(self.test_pid)].copy()
        self.df_playlist_tracks_test = self.df_testset
        self.df_playlist_tracks_count_test = self.df_testset.groupby(['pid', 'tid'], as_index=False)['rating'].count()      
        self.df_playlist_tracks_true = self.df_trueset.sort_values(by=['pid','pos'])
        
        # trainset: part of rawdata + testset
        print("****** 生成训练集 ******")
        print("* 将剩余的数据集与测试集选取的数据合并为训练集\n")
        self.df_playlists_info_train = self.df_playlists_info_copy.copy()
        
        df_pure_train = self.df_playlist_tracks.loc[~self.df_playlist_tracks['pid'].isin(self.test_pid)]
        self.df_playlist_tracks_train = df_pure_train.append(self.df_testset.loc[~self.df_testset['pid'].isin(self.zero_seed_pid)])
        self.df_playlist_tracks_count_train = self.df_playlist_tracks_train.groupby(['pid', 'tid'], as_index=False)['rating'].count()
        
        print("****** 完成 ******")        

          
    def jsonToCSV(self, dataset_type):
        
        if dataset_type == 1:       # rawdata_set
        
            csv_path = "../data/csv/rawdata/"
            
            self.df_playlists_info.to_csv(csv_path + 'playlists_info.csv', index=None)
            self.df_tracks.to_csv(csv_path + 'tracks.csv', index=None)
            self.df_playlist_tracks_count.to_csv(csv_path + 'playlist_tracks.csv')
            
        elif dataset_type == 2:     # test_train_set
        
            csv_path = "../data/csv/real/"
            
            # trainset
            self.df_playlists_info_train.to_csv(csv_path + 'trainset/playlists_info.csv', index=None)
            self.df_tracks.to_csv(csv_path + 'trainset/tracks.csv', index=None)
            self.df_playlist_tracks_count_train.to_csv(csv_path + 'trainset/playlist_tracks.csv')
            
            
            # testset
            self.df_playlists_info_test.to_csv(csv_path + 'testset/playlists_info.csv', index=None)
            self.df_playlist_tracks_count_test.to_csv(csv_path + 'testset/playlist_tracks.csv', index=None)
            self.df_playlist_tracks_true.to_csv(csv_path + 'testset/playlist_tracks_true.csv', index=None)
        
    def jsonToHDF5(self, dataset_type):
        
        if dataset_type == 1:       # rawdata_set
        
            hdf5_path = "../data/hdf5/rawdata/"
            # plyliscts_info
            vaex_df = vaex.from_pandas(self.df_playlists_info, copy_index=False)
            vaex_df.export_hdf5(hdf5_path+'playlists_info.hdf5')
            # tracks
            vaex_df = vaex.from_pandas(self.df_tracks, copy_index=False)
            vaex_df.export_hdf5(hdf5_path+'tracks.hdf5') 
            # playlist_tracks
            vaex_df = vaex.from_pandas(self.df_playlist_tracks_count, copy_index=False)
            vaex_df.export_hdf5(hdf5_path+'playlist_tracks.hdf5')
        elif dataset_type == 2:     # test_train_set
        
            hdf5_path = '../data/hdf5/real/'
            
            # train
            
            # playlists_info
            vaex_df = vaex.from_pandas(self.df_playlists_info_train, copy_index=False)
            vaex_df.export_hdf5(hdf5_path+'trainset/playlists_info.hdf5')
            # tracks
            vaex_df = vaex.from_pandas(self.df_tracks, copy_index=False)
            vaex_df.export_hdf5(hdf5_path+'tracks.hdf5') 
            # playlists_tracks
            vaex_df = vaex.from_pandas(self.df_playlist_tracks_count_train, copy_index=False)
            vaex_df.export_hdf5(hdf5_path+'trainset/playlist_tracks.hdf5')
            
            # testset
            
            # playlists_info
            vaex_df = vaex.from_pandas(self.df_playlists_info_test, copy_index=False)
            vaex_df.export_hdf5(hdf5_path+'testset/playlists_info.hdf5')
            
            # playlist_tracks
            vaex_df = vaex.from_pandas(self.df_playlist_tracks_count_test, copy_index=False)
            vaex_df.export_hdf5(hdf5_path+'testset/playlist_tracks.hdf5')
            
            # playlist_tracks_true
            vaex_df = vaex.from_pandas(self.df_playlist_tracks_true, copy_index=False)
            vaex_df.export_hdf5(hdf5_path+'testset/playlist_tracks_true.hdf5')
            

        

        
    




    
