import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader
import pandas
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
import glob
from PIL import Image

import IPython
e = IPython.embed

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            qvel = root['/observations/qvel'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
            # get all actions after and including start_ts
            if is_sim:
                action = root['/action'][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root['/action'][max(0, start_ts - 1):] # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

        self.is_sim = is_sim
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            qvel = root['/observations/qvel'][()]
            action = root['/action'][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


class SegmentType(Enum):
    ACTION = 'action'
    OBSERVATION = 'observation'
    BOTH = 'both'
    

class TargetType(Enum):
    POS = 'qpos'
    VELOCITY = 'qvel'

@dataclass
class QuatSegment:
    length: int
    type: SegmentType
    target: TargetType


def quat2h5(data_folder: str, quat_file_name: str, 
            h5_file: str, quat_cfg: OrderedDict[str, QuatSegment],
            image_ext: str = 'png', cam_name='wrist', max_image_height=480)-> None:
    """Convert the data and the processed sequence into ACT h5 file
    Args:
        data_folder: the folder containing the data
        quat_file_name: the name of the file containing the quaternions
        h5_file: the name of the h5 file to be saved
        quat_cfg: the configuration of the quaternions
    """
    quat_path = os.path.join(data_folder, quat_file_name)
    df = pandas.read_csv(quat_path, header=None, sep=' ').iloc[:, :-1] # remove the last column (NaN)
    
    # quat cfg should lead the same columns as the data
    total_columns = sum([v.length for v in quat_cfg.values()]) + 1 # 1 for ts
    assert total_columns == df.shape[1], f"Total columns in quat_cfg ({total_columns}) should match the number of columns in the data ({df.shape[1]})"
    
    with h5py.File(h5_file, 'w') as f:
        # create dataset arrays
        qpos = []
        qvel = []
        action = []
        image_files = []
        ptr = 1
        for k, v in quat_cfg.items():
            v_data = df.iloc[:, ptr:ptr+v.length].to_numpy()
            # move the pointer
            ptr += v.length
            
            # add data to their locations
            if v.type == SegmentType.ACTION or v.type == SegmentType.BOTH:
                action.append(v_data)
                
            if v.type == SegmentType.OBSERVATION or v.type == SegmentType.BOTH:
                if v.target == TargetType.POS:
                    qpos.append(v_data)
                elif v.target == TargetType.VELOCITY:
                    qvel.append(v_data)
                else:
                    raise ValueError(f"Invalid target type {v.target}")
        
        qpos = np.concatenate(qpos, axis=1).astype(np.float32)
        qvel = np.concatenate(qvel, axis=1).astype(np.float32)
        action = np.concatenate(action, axis=1).astype(np.float32)
        print(f"qpos shape: {qpos.shape}, qvel shape: {qvel.shape}, action shape: {action.shape}")
        
        f.create_dataset('observations/qpos', data=qpos)
        f.create_dataset('observations/qvel', data=qvel)
        f.create_dataset('action', data=action)
        
        # process images
        data_ts = df.iloc[:, 0].to_numpy()
        image_files = glob.glob(os.path.join(data_folder, f'*.{image_ext}'))
        
        def find_closest_image(data_ts, image_ts):
            diff = np.abs(data_ts[:, None] - image_ts[None, :])
            idx = np.argmin(diff, axis=1)
            min_diff = np.min(diff, axis=1)
            return idx, min_diff
        
        image_ts = [float(os.path.basename(f).split('.')[0].split('_')[-1]) for f in image_files]
        image_idx, time_diff = find_closest_image(data_ts, np.array(image_ts))
        
        all_images = [Image.open(p) for p in image_files]
        resized_images = []
        print(f"resizing {len(all_images)} images")
        for img in all_images:
            w, h = img.size
            height_percent = (max_image_height / float(h))
            width_size = int((float(w) * float(height_percent)))
            resized_image = np.array(img.resize((width_size, max_image_height), resample=Image.Resampling.BILINEAR))
            resized_images.append(resized_image)
        all_images = resized_images
        print("done resizing images")
        image_series = np.array([all_images[i] for i in image_idx])
        f.create_dataset(f'observations/images/{cam_name}', data=image_series)
        f.create_dataset(f'observations/images/{cam_name}_time_diff', data=time_diff)
        
        # no need for aligning step in dataloader
        f.attrs['sim'] = 1
        
        print(f"data saved to {h5_file}")
        
        
if __name__ == "__main__":
    # test quat2h5
    quat_cfg = OrderedDict([
        ('traj', QuatSegment(length=7, type=SegmentType.BOTH, target=TargetType.POS)),
        ('handjs', QuatSegment(length=6, type=SegmentType.BOTH, target=TargetType.POS)),
        ('handvel', QuatSegment(length=6, type=SegmentType.OBSERVATION, target=TargetType.VELOCITY)),
        ('handforce', QuatSegment(length=6, type=SegmentType.ACTION, target=TargetType.POS)),
    ])
    
    data_folder = '/home/yjxiong/act/sim_data/artly/hand_recording/11_37_05'
    quat_file_name = 'place_cup_processed_recorded_seq1_11_37_05.quat'
    h5_file = 'episode_test.hdf5'
    
    quat2h5(data_folder, quat_file_name, h5_file, quat_cfg)