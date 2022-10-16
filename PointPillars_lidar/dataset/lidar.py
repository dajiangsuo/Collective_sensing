import numpy as np
import os
import torch
from torch.utils.data import Dataset

import sys
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE))

from utils import read_pickle, read_points, bbox_camera2lidar
from dataset import point_range_filter, data_augment


class BaseSampler():
    def __init__(self, sampled_list, shuffle=True):
        self.total_num = len(sampled_list)
        self.sampled_list = np.array(sampled_list)
        self.indices = np.arange(self.total_num)
        if shuffle:
            np.random.shuffle(self.indices)
        self.shuffle = shuffle
        self.idx = 0

    def sample(self, num):
        if self.idx + num < self.total_num:
            ret = self.sampled_list[self.indices[self.idx:self.idx+num]]
            self.idx += num
        else:
            ret = self.sampled_list[self.indices[self.idx:]]
            self.idx = 0
            if self.shuffle:
                np.random.shuffle(self.indices)
        return ret


class LIDAR(Dataset):

    CLASSES = {
        'Car': 0
        }

    def __init__(self, data_root, split, pts_prefix='lidar'):
        assert split in ['train', 'val', 'trainval', 'test']
        self.data_root = data_root
        self.split = split
        self.pts_prefix = pts_prefix
        self.data_infos = read_pickle(os.path.join(data_root, f'lidar_infos_{split}.pkl'))
        self.sorted_ids = list(self.data_infos.keys())
        db_infos = read_pickle(os.path.join(data_root, 'lidar_dbinfos_train.pkl'))

        db_sampler = {}
        for cat_name in self.CLASSES:
            db_sampler[cat_name] = BaseSampler(db_infos[cat_name], shuffle=True)
        self.data_aug_config=dict(
            db_sampler=dict(
                db_sampler=db_sampler,
                sample_groups=dict(Car=15)
                ),
            object_noise=dict(
                num_try=100,
                translation_std=[0.25],
                rot_range=[-0.15707963267, 0.15707963267]
                ),
            random_flip_ratio=0.5,
            global_rot_scale_trans=dict(
                rot_range=[-0.78539816, 0.78539816],
                scale_ratio_range=[0.95, 1.05],
                translation_std=[0, 0, 0]
                ), 
            point_range_filter=[0, -39.68, -3, 69.12, 39.68, 1],
            object_range_filter=[0, -39.68, -3, 69.12, 39.68, 1]             
        )

    # def filter_db(self, db_infos):
    #     # 1. filter_by_difficulty
    #     for k, v in db_infos.items():
    #         db_infos[k] = [item for item in v if item['difficulty'] != -1]

    #     # 2. filter_by_min_points, dict(Car=5, Pedestrian=10, Cyclist=10)
    #     filter_thrs = dict(Car=5, Pedestrian=10, Cyclist=10)
    #     for cat in self.CLASSES:
    #         filter_thr = filter_thrs[cat]
    #         db_infos[cat] = [item for item in db_infos[cat] if item['num_points_in_gt'] >= filter_thr]
        
    #     return db_infos

    def __getitem__(self, index):
        data_info = self.data_infos[self.sorted_ids[index]]

        # point cloud input
        pts_path = data_info['lidar']
        pts = read_points(pts_path)

        # annotations input
        annos_info = data_info['annos']
        annos_name = annos_info['name']
        annos_location = annos_info['location']
        annos_dimension = annos_info['dimensions']
        rotation_y = annos_info['rotation_y']
        gt_bboxes_3d = np.concatenate([annos_location, annos_dimension, rotation_y[:, None]], axis=1).astype(np.float32)
        #gt_bboxes_3d = bbox_camera2lidar(gt_bboxes, None, None)
        gt_labels = [self.CLASSES.get(name, -1) for name in annos_name]
        data_dict = {
            'pts': pts,
            'gt_bboxes_3d': gt_bboxes_3d,
            'gt_labels': np.array(gt_labels), 
            'gt_names': annos_name,
            'difficulty': annos_info['difficulty'],
            # 'image_info': image_info,
            'index': self.sorted_ids[index],
            # 'calib_info': calib_info
        }
        if self.split in ['train', 'trainval']:
            data_dict = data_augment(self.CLASSES, self.data_root, data_dict, self.data_aug_config)
        else:
            data_dict = point_range_filter(data_dict, point_range=self.data_aug_config['point_range_filter'])
        data_dict['index'] = self.sorted_ids[index]
        return data_dict

    def __len__(self):
        return len(self.data_infos)

