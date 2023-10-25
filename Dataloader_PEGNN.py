import torch
import os
import random
import pickle
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from osgeo import gdal
import concurrent.futures


class IntpDataset(Dataset):
    def __init__(self, settings, mask_distance, call_name):
        # save init parameters
        self.origin_path = settings['origin_path']
        self.time_fold = settings['fold']
        self.holdout = settings['holdout']
        self.mask_distance = mask_distance
        self.lowest_rank = settings['lowest_rank']
        self.call_name = call_name
        self.debug = settings['debug']
        
        # load norm info
        with open(self.origin_path + f'Folds_Info/norm_{self.time_fold}.info', 'rb') as f:
            self.dic_op_minmax, self.dic_op_meanstd = pickle.load(f)
            
        # Generate call_scene_list from 'divide_set_{self.time_fold}.info', then generate call_list from call_scene_list
        with open(self.origin_path + f'Folds_Info/divide_set_{self.time_fold}.info', 'rb') as f:
            divide_set = pickle.load(f)
        if call_name == 'train':
            # for training set, call_list is same as call_scene_list, because call item will be chosen randomly
            call_scene_list = divide_set[0]
            if self.debug and len(call_scene_list) > 2000:
                call_scene_list = call_scene_list[:2000]
            # do normalization in the parallel fashion
            self.total_df_dict = {}
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(self.process_child, file_name) for file_name in call_scene_list]
                for future in concurrent.futures.as_completed(futures):
                    file_name, file_content = future.result()
                    self.total_df_dict[file_name] = file_content
            self.call_list = []
            for scene in call_scene_list:
                df = self.total_df_dict[scene]
                all_candidate = df[df['op']=='mcpm10']
                for index in all_candidate.index.tolist():
                    for mask_buffer in range(51):
                        self.call_list.append([scene, index, mask_buffer])
        elif call_name == 'test':
            # for Early Stopping set, call_list is 3 label stations excluding holdout
            call_scene_list = divide_set[1]
            if self.debug and len(call_scene_list) > 1000:
                call_scene_list = call_scene_list[:1000]
            # do normalization in the parallel fashion
            self.total_df_dict = {}
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(self.process_child, file_name) for file_name in call_scene_list]
                for future in concurrent.futures.as_completed(futures):
                    file_name, file_content = future.result()
                    self.total_df_dict[file_name] = file_content
            if 'Dataset_res250' in self.origin_path:
                stations = [0, 1, 2, 3]
                stations.remove(self.holdout)
            elif 'LUFT_res250' in self.origin_path:
                label_stations = [0, 1, 2, 3, 4, 5]
                stations = [x for x in label_stations if x not in self.holdout]
            self.call_list = []
            for scene in call_scene_list:
                for station in stations:
                    self.call_list.append([scene, station])
        elif call_name == 'eval':
            # for Final Evaluation set, call_list is holdout station
            call_scene_list = divide_set[1]
            if self.debug and len(call_scene_list) > 1000:
                call_scene_list = call_scene_list[:1000]
            # do normalization in the parallel fashion
            self.total_df_dict = {}
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(self.process_child, file_name) for file_name in call_scene_list]
                for future in concurrent.futures.as_completed(futures):
                    file_name, file_content = future.result()
                    self.total_df_dict[file_name] = file_content
            self.call_list = []
            if 'Dataset_res250' in self.origin_path:
                for scene in call_scene_list:
                    self.call_list.append([scene, self.holdout])
            elif 'LUFT_res250' in self.origin_path:
                for scene in call_scene_list:
                    for station in self.holdout:
                        self.call_list.append([scene, station])
        elif call_name == 'train_self':
            # for training set, call_list is same as call_scene_list, because call item will be chosen randomly
            call_scene_list = divide_set[0]
            if self.debug and len(call_scene_list) > 2000:
                call_scene_list = call_scene_list[:2000]
            # do normalization in the parallel fashion
            self.total_df_dict = {}
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(self.process_child, file_name) for file_name in call_scene_list]
                for future in concurrent.futures.as_completed(futures):
                    file_name, file_content = future.result()
                    self.total_df_dict[file_name] = file_content
            self.call_list = []
            for scene in call_scene_list:
                df = self.total_df_dict[scene]
                all_candidate = df[df['op']=='mcpm10']
                self.call_list.append([scene, all_candidate.index[0]])
                
        print(len(self.total_df_dict.keys()))
        
        # process the land-use tif
        geo_file = gdal.Open(self.origin_path + 'CWSL_norm.tif')
        tif_channel_list = []
        for i in range(geo_file.RasterCount):
            tif_channel_list.append(np.array(geo_file.GetRasterBand(i + 1).ReadAsArray(), dtype="float32"))
        self.tif = torch.from_numpy(np.stack(tif_channel_list, axis=0))
        self.width = geo_file.RasterXSize
        self.height = geo_file.RasterYSize
        
        # list interested ops
        self.op_dic = {'mcpm10': 0, 'mcpm2p5': 1, 'ta': 2, 'hur': 3, 'plev': 4, 'precip': 5, 'wsx': 6, 'wsy': 7, 'globalrad': 8, }
        self.possible_values = list(self.op_dic.keys())
        self.possible_values.sort()
        
    def __len__(self):
        return len(self.call_list)
    
    def process_child(self, filename):
        df = pd.read_csv(self.origin_path + 'Dataset_Separation/' + filename, sep=';')
        # drop everything in bad quality
        df = df[df['Thing']>=self.lowest_rank]
        # normalize all values (coordinates will be normalized later)
        df = self.norm(d=df)
        
        return filename, df
    
    def norm(self, d):
        d_list = []
        for op in d['op'].unique():
            d_op = d.loc[d['op']==op].copy(deep=True)
            if op in ['s_label_0', 's_label_1', 's_label_2', 's_label_3', 's_label_4', 's_label_5', 's_label_6', 'p_label']:
                op_norm = 'mcpm10'
            else:
                op_norm = op
            if op_norm in self.dic_op_minmax.keys():
                d_op['Result_norm'] = (d_op['Result'] - self.dic_op_minmax[op_norm][0]) / (self.dic_op_minmax[op_norm][1] - self.dic_op_minmax[op_norm][0])
            elif op_norm in self.dic_op_meanstd.keys():
                d_op['Result_norm'] = (d_op['Result'] - self.dic_op_meanstd[op_norm][0]) / self.dic_op_meanstd[op_norm][1]
            d_list.append(d_op)
        return pd.concat(d_list, axis=0, ignore_index=False)
    
    def distance_matrix(self, x0, y0, x1, y1):
        obs = np.vstack((x0, y0)).T
        interp = np.vstack((x1, y1)).T
        d0 = np.subtract.outer(obs[:,0], interp[:,0])
        d1 = np.subtract.outer(obs[:,1], interp[:,1])
        # calculate hypotenuse
        return np.hypot(d0, d1)
    
    def idw_interpolation(self, x, y, values, xi, yi, p=2):
        dist = self.distance_matrix(x, y, xi,yi)
        # In IDW, weights are 1 / distance
        weights = 1.0/(dist+1e-12)**p
        # Make weights sum to one
        weights /= weights.sum(axis=0)
        # Multiply the weights for each interpolated point by all observed Z-values
        return np.dot(weights.T, values)
            
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # get call_item and scenario_story
        if self.call_name == 'train':
            # for training set, call item is randomly taken from the 'mcpm10'
            #     - get the scenario, filter with lowest reliable sensor rank
            scene = self.call_list[idx][0]
            random_index = self.call_list[idx][1]
            
            df = self.total_df_dict[scene]
            #     - get a random call_item, those remianed are story
            all_candidate = df[df['op']=='mcpm10']

            random_row = all_candidate.loc[random_index]
            call_item = pd.DataFrame([random_row], index=[random_index])
            remaining_candidate = all_candidate.drop(random_index)
            rest_story = df.loc[(df['op'].isin(self.op_dic.keys())) & (df['op']!='mcpm10')]
            df_story = pd.concat([remaining_candidate, rest_story], axis=0, ignore_index=True)
            
        elif self.call_name in ['test', 'eval']:
            # for Early Stopping set and Final Evaluation set
            #     - get the scenario and target station, filter with lowest reliable sensor rank
            scene = self.call_list[idx][0]
            target = self.call_list[idx][1]
            df = self.total_df_dict[scene]
            #     - get call_item by target station, story are filtered with op_dic
            all_candidate = df[df['op']=='mcpm10']
            
            call_item = df[df['op']==f's_label_{target}']
            if len(call_item) != 1:
                call_item = call_item[0]
            df_story = df.loc[df['op'].isin(self.op_dic.keys())]
            
        elif self.call_name == 'train_self':
            # for training set, call item is randomly taken from the 'mcpm10'
            #     - get the scenario, filter with lowest reliable sensor rank
            scene = self.call_list[idx][0]
            random_index = self.call_list[idx][1]
            
            df = self.total_df_dict[scene]
            #     - get a random call_item, those remianed are story
            all_candidate = df[df['op']=='mcpm10']

            random_row = all_candidate.loc[random_index]
            call_item = pd.DataFrame([random_row], index=[random_index])
            remaining_candidate = all_candidate.drop(random_index)
            rest_story = df.loc[(df['op'].isin(self.op_dic.keys())) & (df['op']!='mcpm10')]
            df_story = pd.concat([remaining_candidate, rest_story], axis=0, ignore_index=True)
        
        # processing senario informations:
        #     - mask out all readings within 'mask_distance'
        if self.mask_distance == -1:
            this_mask = self.call_list[idx][2]
        else:
            this_mask = self.mask_distance
        df_filtered = df_story.loc[(abs(df_story['Longitude'] - call_item.iloc[0, 2]) + abs(df_story['Latitude'] - call_item.iloc[0, 3])) >= this_mask, :].copy()
        df_filtered = df_filtered.drop(columns=['Result'])
        
        #     - generate story token serie [value, rank, loc_x, loc_y, op]
        df_filtered = df_filtered[['Result_norm', 'Thing', 'Longitude', 'Latitude', 'op']]
        aggregated_df = df_filtered.groupby(['Longitude', 'Latitude', 'op']).mean().reset_index()
        
        
        # get gragh candidates
        if self.call_name in ['train', 'train_self']:
            graph_candidates = all_candidate[['Result_norm', 'Thing', 'Longitude', 'Latitude', 'op']].copy()
        elif self.call_name in ['test', 'eval']:
            graph_candidates_1 = df[df['op'] == f's_label_{self.call_list[idx][1]}'][['Result_norm', 'Thing', 'Longitude', 'Latitude', 'op']].copy()
            # graph_candidates_1 = aggregated_df[aggregated_df['op'] == f's_label_{self.call_list[idx][1]}']
            # print(len(graph_candidates_1))
            graph_candidates_2 = all_candidate[['Result_norm', 'Thing', 'Longitude', 'Latitude', 'op']].copy()
            graph_candidates = pd.concat([graph_candidates_1, graph_candidates_2], axis=0)
            
        # print(graph_candidates)
        coords = torch.from_numpy(graph_candidates[['Longitude', 'Latitude']].values).float()
        answers = torch.from_numpy(graph_candidates[['Result_norm']].values).float()
        
        features = torch.zeros((len(graph_candidates), len(self.op_dic) + 1))
        # print(features.size())
        for op in self.possible_values:
            aggregated_df_op = aggregated_df[aggregated_df['op']==op]
            interpolated_grid = torch.zeros((len(graph_candidates), 1))
            if len(aggregated_df_op) != 0:
                xi = graph_candidates['Longitude'].values
                yi = graph_candidates['Latitude'].values
                x = aggregated_df_op['Longitude'].values
                y = aggregated_df_op['Latitude'].values
                values = aggregated_df_op['Result_norm'].values
                interpolated_values = self.idw_interpolation(x, y, values, xi, yi)
                interpolated_grid = torch.from_numpy(interpolated_values).reshape((len(graph_candidates), 1))
                # print(interpolated_grid.size())
            features[:, self.op_dic[op]:self.op_dic[op]+1] = interpolated_grid
        
        conditions = (graph_candidates['op'] == 'mcpm10') | (graph_candidates['op'] == f's_label_{self.call_list[idx][1]}')
        features[:, -1] = torch.from_numpy(np.where(conditions, graph_candidates['Thing'] / 3, (self.lowest_rank-1)/3))
        features = features.float()
        
        return features, coords, answers

    
def collate_fn(examples):
    input_lenths = torch.tensor([len(ex[0]) for ex in examples if len(ex[0]) > 2])
    x_b = pad_sequence([ex[0] for ex in examples if len(ex[0]) > 2], batch_first=True, padding_value=0.0)
    c_b = pad_sequence([ex[1] for ex in examples if len(ex[1]) > 2], batch_first=True, padding_value=0.0)
    y_b = pad_sequence([ex[2] for ex in examples if len(ex[2]) > 2], batch_first=True, padding_value=0.0)
    return x_b, c_b, y_b, input_lenths