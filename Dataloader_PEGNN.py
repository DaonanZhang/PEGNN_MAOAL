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


def encode_and_bind(original_dataframe, feature_to_encode, possible_values):
    enc_df = original_dataframe.copy()
    for value in possible_values:
        enc_df.loc[:, feature_to_encode + '_' + str(value)] = (enc_df[feature_to_encode] == value).astype(int)
    res = enc_df.drop([feature_to_encode], axis=1)
    return res


class IntpDataset(Dataset):
    def __init__(self, settings, mask_distance, call_name):
        # save init parameters
        self.origin_path = settings['origin_path']
        self.time_fold = settings['fold']
        self.holdout = settings['holdout']

        self.mask_distance = mask_distance

        # set lowest_rank in order to generate the k-nearest neighbors?
        self.lowest_rank = settings['lowest_rank']
        self.call_name = call_name
        self.debug = settings['debug']
        self.aux_task_num = settings['aux_task_num']

        # load norm info, std, mean
        with open(self.origin_path + f'Folds_Info/norm_{self.time_fold}.info', 'rb') as f:
            self.dic_op_minmax, self.dic_op_meanstd = pickle.load(f)
        f.close()

        # Generate call_scene_list from 'divide_set_{self.time_fold}.info', then generate call_list from call_scene_list
        with open(self.origin_path + f'Folds_Info/divide_set_{self.time_fold}.info', 'rb') as f:
            divide_set = pickle.load(f)
        f.close()

        if call_name == 'train':
            # for training set, call_list is same as call_scene_list, because call item will be chosen randomly
            call_scene_list = divide_set[0]
            # list of CSV files

            # debug mode
            if self.debug and len(call_scene_list) > 2000:
                call_scene_list = call_scene_list[:2000]

            # do normalization in the parallel fashion
            # drop bad quality and normalize
            self.total_df_dict = {}
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # return filename, df in the process_child function
                futures = [executor.submit(self.process_child, file_name) for file_name in call_scene_list]
                for future in concurrent.futures.as_completed(futures):
                    file_name, file_content = future.result()
                    # save the result in file_content
                    self.total_df_dict[file_name] = file_content

            self.call_list = []
            for scene in call_scene_list:
                # df = accordingly scene DataFrame
                df = self.total_df_dict[scene]
                all_candidate = df[df['op'] == 'mcpm10']
                for index in all_candidate.index.tolist():
                    for mask_buffer in range(51):
                        self.call_list.append([scene, index, mask_buffer])

        #     each csv file 52 times mask_buffer

        # for test_set
        elif call_name == 'test':
            # for Early Stopping set, call_list is 3 label stations excluding holdout
            # divide_set[0] is train set, divide_set[1] is test set
            call_scene_list = divide_set[1]

            # debug mode
            if self.debug and len(call_scene_list) > 1000:
                call_scene_list = call_scene_list[:1000]

            # do normalization in the parallel fashion
            self.total_df_dict = {}
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(self.process_child, file_name) for file_name in call_scene_list]
                for future in concurrent.futures.as_completed(futures):
                    file_name, file_content = future.result()
                    self.total_df_dict[file_name] = file_content

            # fold_out
            stations = [0, 1, 2, 3]
            if 'Dataset_res250' in self.origin_path:
                for station in self.holdout:
                    stations.remove(station)

            self.call_list = []
            for scene in call_scene_list:
                for station in stations:
                    self.call_list.append([scene, station])

        # for eval_set
        elif call_name == 'eval':
            # for Final Evaluation set, call_list is holdout station
            call_scene_list = divide_set[2]

            if self.debug and len(call_scene_list) > 1000:
                call_scene_list = call_scene_list[:1000]

            # do normalization in the parallel fashion
            self.total_df_dict = {}
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(self.process_child, file_name) for file_name in call_scene_list]

                # each time when a thread/process is done, it would be immediately return as future in the loop
                for future in concurrent.futures.as_completed(futures):
                    file_name, file_content = future.result()
                    self.total_df_dict[file_name] = file_content

            # eval
            stations = []
            if 'Dataset_res250' in self.origin_path:
                for station in self.holdout:
                    stations.append(station)

            self.call_list = []
            for scene in call_scene_list:
                for station in stations:
                    self.call_list.append([scene, station])

        elif call_name == 'aux':

            call_scene_list = divide_set[3]

            # debug mode
            if self.debug and len(call_scene_list) > 2000:
                call_scene_list = call_scene_list[:2000]

            # do normalization in the parallel fashion
            # drop bad quality and normalize
            self.total_df_dict = {}
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # return filename, df in the process_child function
                futures = [executor.submit(self.process_child, file_name) for file_name in call_scene_list]
                for future in concurrent.futures.as_completed(futures):
                    file_name, file_content = future.result()
                    # save the result in file_content
                    self.total_df_dict[file_name] = file_content

            self.call_list = []
            for scene in call_scene_list:
                # df = accordingly scene DataFrame
                df = self.total_df_dict[scene]
                all_candidate = df[df['op'] == 'mcpm10']
                for index in all_candidate.index.tolist():
                    for mask_buffer in range(51):
                        self.call_list.append([scene, index, mask_buffer])

        geo_file = gdal.Open(self.origin_path + 'CWSL_norm.tif')
        tif_channel_list = []
        for i in range(geo_file.RasterCount):
            tif_channel_list.append(np.array(geo_file.GetRasterBand(i + 1).ReadAsArray(), dtype="float32"))
        self.tif = torch.from_numpy(np.stack(tif_channel_list, axis=0))
        self.width = geo_file.RasterXSize
        self.height = geo_file.RasterYSize

        # only keep the mcmp10 and thing_class
        self.primary_op = {'mcpm10': 0}
        self.primary_op_list = list(self.primary_op.keys())
        self.primary_op_list.sort()

        self.aux_op_dic = settings['aux_op_dic']
        self.aux_op_list = list(self.aux_op_dic.keys())
        self.aux_op_list.sort()

        self.env_op_dic = settings['env_op_dic']
        self.env_op_list = list(self.env_op_dic.keys())
        self.env_op_list.sort()

    def __len__(self):
        return len(self.call_list)

    # task for each process/thread
    #   - read the csv file
    #  - drop all bad quality readings
    def process_child(self, filename):
        file_path = self.origin_path + 'Dataset_Separation/' + filename
        with open(file_path, 'r') as file:
            df = pd.read_csv(file, sep=';')
            # drop everything in bad quality
            # loweset_rank = quality level of the sensor
            df = df[df['Thing'] >= self.lowest_rank]
            # normalize all values (coordinates will be normalized later)
            df = self.norm(d=df)
        file.close()
        return filename, df

    # def process_child(self, filename):
    #     df = pd.read_csv(self.origin_path + 'Dataset_Separation/' + filename, sep=';')
    #     # drop everything in bad quality
    #     # loweset_rank = quality level of the sensor
    #     df = df[df['Thing'] >= self.lowest_rank]
    #     # normalize all values (coordinates will be normalized later)
    #     df = self.norm(d=df)
    #     return filename, df

    # normalize all values and the thing
    def norm(self, d):
        d_list = []
        for op in d['op'].unique():
            d_op = d.loc[d['op'] == op].copy(deep=True)
            if op in ['s_label_0', 's_label_1', 's_label_2', 's_label_3', 's_label_4', 's_label_5', 's_label_6',
                      'p_label']:
                op_norm = 'mcpm10'
            else:
                op_norm = op
            # do the normalization
            if op_norm in self.dic_op_minmax.keys():
                d_op['Result_norm'] = (d_op['Result'] - self.dic_op_minmax[op_norm][0]) / (
                            self.dic_op_minmax[op_norm][1] - self.dic_op_minmax[op_norm][0])

            elif op_norm in self.dic_op_meanstd.keys():
                d_op['Result_norm'] = (d_op['Result'] - self.dic_op_meanstd[op_norm][0]) / self.dic_op_meanstd[op_norm][
                    1]

            # norm the thing
            d_op['Thing'] /= 3

            d_list.append(d_op)
        return pd.concat(d_list, axis=0, ignore_index=False)

    def __getitem__(self, idx):

        # TODO: Debug
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get call_item and scenario_story
        if self.call_name == 'train':
            # for training set, call item is randomly taken from the 'mcpm10'
            #     - get the scenario, filter with lowest reliable sensor rank
            scene = self.call_list[idx][0]
            random_index = self.call_list[idx][1]

            # df is a directory of all dataframes
            df = self.total_df_dict[scene]
            #     - get a random call_item, those remained are story
            all_candidate = df[df['op'] == 'mcpm10']

            # only get random set for training set
            random_row = all_candidate.loc[random_index]
            call_item = pd.DataFrame([random_row])
            remaining_candidate = all_candidate.drop(random_index)

            # only pm10 
            df_story = remaining_candidate
            aux_story = df.loc[(df['op'].isin(self.aux_op_dic.keys()))]
            rest_story = df.loc[(df['op'].isin(self.env_op_dic.keys()))]

        elif self.call_name == 'aux':
            # for training set, call item is randomly taken from the 'mcpm10'
            #     - get the scenario, filter with lowest reliable sensor rank
            scene = self.call_list[idx][0]
            random_index = self.call_list[idx][1]

            # df is a directory of all dataframes
            df = self.total_df_dict[scene]
            #     - get a random call_item, those remained are story
            all_candidate = df[df['op'] == 'mcpm10']

            # only get random set for training set
            random_row = all_candidate.loc[random_index]
            call_item = pd.DataFrame([random_row])
            remaining_candidate = all_candidate.drop(random_index)

            # only pm10
            df_story = remaining_candidate
            aux_story = df.loc[(df['op'].isin(self.aux_op_dic.keys()))]
            rest_story = df.loc[(df['op'].isin(self.env_op_dic.keys()))]

        elif self.call_name in ['test', 'eval']:
            # for Early Stopping set and Final Evaluation set
            #     - get the scenario and target station, filter with lowest reliable sensor rank
            scene = self.call_list[idx][0]
            target = self.call_list[idx][1]
            df = self.total_df_dict[scene]
            #     - get call_item by target station, story are filtered with op_dic
            # get the target station

            call_item = df[df['op'] == f's_label_{target}']
            if len(call_item) != 1:
                call_item = call_item[0]

            df_story = df.loc[df['op'].isin(self.primary_op.keys())]
            aux_story = df.loc[(df['op'].isin(self.aux_op_dic.keys()))]
            rest_story = df.loc[(df['op'].isin(self.env_op_dic.keys()))]

            # ______________________________train_self no usage_____________________________________________
        # elif self.call_name == 'train_self':
        #     # for training set, call item is randomly taken from the 'mcpm10'
        #     #     - get the scenario, filter with lowest reliable sensor rank
        #     scene = self.call_list[idx][0]
        #     random_index = self.call_list[idx][1]
        #
        #     df = self.total_df_dict[scene]
        #     #     - get a random call_item, those remianed are story
        #     all_candidate = df[df['op']=='mcpm10']
        #     random_row = all_candidate.loc[random_index]
        #
        #     call_item = pd.DataFrame([random_row], index=[random_index])
        #     remaining_candidate = all_candidate.drop(random_index)
        #
        #     df_story = remaining_candidate
        #     aux_story = df.loc[(df['op'].isin(self.aux_op_dic.keys()))]
        #     rest_story = df.loc[(df['op'].isin(self.env_op_dic.keys()))]

        # processing scenario information:
        #     - mask out all readings within 'mask_distance'
        if self.mask_distance == -1:
            this_mask = self.call_list[idx][2]
        else:
            this_mask = self.mask_distance

        # _________________________________________aggregation_______________________________________________________
        # only pm10

        df_filtered = df_story.loc[(abs(df_story['Longitude'] - call_item.iloc[0, 2]) + abs(
            df_story['Latitude'] - call_item.iloc[0, 3])) >= this_mask, :].copy()
        df_filtered = df_filtered.drop(columns=['Result'])
        #     - generate story token serie [value, rank, loc_x, loc_y, op]
        df_filtered = df_filtered[['Result_norm', 'Thing', 'Longitude', 'Latitude', 'op']]
        aggregated_df = df_filtered.groupby(['Longitude', 'Latitude', 'op']).mean().reset_index()

        # other aux_pms
        aux_filtered = aux_story.drop(columns=['Result'])
        # - generate story token serie [value, rank, loc_x, loc_y, op]
        aux_filtered = aux_filtered[['Result_norm', 'Thing', 'Longitude', 'Latitude', 'op']]
        aggregated_aux = aux_filtered.groupby(['Longitude', 'Latitude', 'op']).mean().reset_index()

        # # processing rest features:
        env_filtered = rest_story.drop(columns=['Result'])
        #     - generate story token serie [value, rank, loc_x, loc_y, op]
        env_filtered = env_filtered[['Result_norm', 'Thing', 'Longitude', 'Latitude', 'op']]
        aggregated_env = env_filtered.groupby(['Longitude', 'Latitude', 'op']).mean().reset_index()

        # _________________________________________get task graph_candidate for target pm_______________________________________________________
        # if self.call_name in ['train', 'train_self']:
        #     graph_candidates = aggregated_df[['Result_norm', 'Thing', 'Longitude', 'Latitude', 'op']].copy()
        #
        # elif self.call_name in ['test', 'eval']:
        #     graph_candidates_1 = aggregated_df[aggregated_df['op'] == f's_label_{self.call_list[idx][1]}'][
        #         ['Result_norm', 'Thing', 'Longitude', 'Latitude', 'op']].copy()
        #     graph_candidates_2 = aggregated_df[['Result_norm', 'Thing', 'Longitude', 'Latitude', 'op']].copy()
        #     graph_candidates = pd.concat([graph_candidates_1, graph_candidates_2], axis=0)

        # _____________________________________append the call_item in the head of the graph_candidates_______________________________________________________

        graph_candidates1 = call_item[['Result_norm', 'Thing', 'Longitude', 'Latitude', 'op']].copy()
        graph_candidates2 = aggregated_df[['Result_norm', 'Thing', 'Longitude', 'Latitude', 'op']].copy()

        # print(f'graph_candidates1: {graph_candidates1}')

        graph_candidates = pd.concat([graph_candidates1, graph_candidates2], axis=0).reset_index()

        # print(f'graph_candidates: {graph_candidates.iloc[0,:]}')

        # _________________________________________target_task_______________________________________________________
        #     - get pm10's coordinates and answers
        coords = torch.from_numpy(graph_candidates[['Longitude', 'Latitude']].values).float()

        # only the call_item's result
        answers = torch.from_numpy(graph_candidates[['Result_norm']].values).float()

        # same sequence
        features_df = graph_candidates[['Result_norm', 'Thing']]
        # only the result of (norm_value of mcpm10/s_label, thing/3)
        features = torch.from_numpy(features_df.to_numpy()).float()

        # _________________________________________aux_task_______________________________________________________
        # (#aux_task, norm_value)
        aux_answers = [torch.zeros(len(graph_candidates), dtype=torch.float) for _ in range(self.aux_task_num)]

        if self.call_name not in ['test', 'eval']:
            for op_index, aux_op in enumerate(self.aux_op_list):
                aggregated_aux_op = aggregated_aux[aggregated_aux['op'] == aux_op]
                # only one row
                for features_df_index, row in graph_candidates.iterrows():
                    # log the aux_op and keep the sequence with aggregated_df
                    # print(f'graph_candidates: {row}')
                    # print(f'aggregated_aux_op: {aggregated_aux_op}')
                    # print('-------------------------------')
                    xi = row['Longitude']
                    yi = row['Latitude']

                    # after aggregation, one postion have maximal one value for each aux_op
                    matched_aux = aggregated_aux_op[
                        (aggregated_aux_op['Longitude'] == xi) & (aggregated_aux_op['Latitude'] == yi)]

                    if (not matched_aux.empty):
                        assigned_value = matched_aux['Result_norm'].values[0]
                        aux_answers[op_index][features_df_index] = assigned_value
                    else:
                        # choose -1 as our Masked value
                        mask_value = -float('inf')
                        assigned_value = mask_value
                        aux_answers[op_index][features_df_index] = assigned_value

                    # print(f'{features_df_index} / {len(graph_candidates)}')
                    # print('_____________________________________________')

                    # print(f'------------- \n xi: {xi} yi:{yi} \n value:{assigned_value} {aux_op}')
                    # print('-------------------------------')
        # _________________________________________env_features_______________________________________________________

        df_one_hot = pd.get_dummies(aggregated_env, columns=['op'])
        required_ops = self.env_op_list
        # check if all the other_features are contained, if not insert the corresponding one-hot row

        for op in required_ops:
            if op not in df_one_hot.columns:
                df_one_hot[op] = 0

        # reorder the columns
        non_op_columns = ['Longitude', 'Latitude', 'Result_norm', 'Thing']
        ordered_columns = non_op_columns + required_ops

        df_one_hot = df_one_hot[ordered_columns]

        # make one hot in int form
        df_one_hot *= 1
        env_features = torch.tensor(df_one_hot.values).float()

        return features, coords, answers, aux_answers, env_features


# collate_fn: how samples are batched together
def collate_fn(examples):
    input_lenths = torch.tensor([len(ex[0]) for ex in examples if len(ex[0]) > 2])

    x_b = pad_sequence([ex[0] for ex in examples if len(ex[0]) > 2], batch_first=True, padding_value=0.0)
    c_b = pad_sequence([ex[1] for ex in examples if len(ex[1]) > 2], batch_first=True, padding_value=0.0)
    y_b = pad_sequence([ex[2] for ex in examples if len(ex[2]) > 2], batch_first=True, padding_value=0.0)
    rest_feature = pad_sequence([ex[4] for ex in examples if len(ex[3]) > 2], batch_first=True, padding_value=0.0)

    task_num = len(examples[0][3])
    aux_y_b = []
    for i in range(0, task_num):
        sequence = pad_sequence([ex[3][i] for ex in examples if len(ex[3]) > 2], batch_first=True, padding_value=-1.0)
        aux_y_b.append(sequence.unsqueeze(-1))

    return x_b, c_b, y_b, aux_y_b, input_lenths, rest_feature