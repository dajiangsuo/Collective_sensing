import numpy as np
import os
import pickle
import yaml
import math

def read_pickle(file_path, suffix='.pkl'):
    assert os.path.splitext(file_path)[1] == suffix
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def write_pickle(results, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(results, f)

def read_points2(file_path, dim=4):
    suffix = os.path.splitext(file_path)[1] 
    assert suffix in ['.bin']
    if suffix == '.bin':
        return np.fromfile(file_path, dtype=np.float32).reshape(-1, dim)
    else:
        raise NotImplementedError


# Dajiang changed
def read_points(file_path, dim=4):
    suffix = os.path.splitext(file_path)[1] 
    assert suffix in ['.pcd', '.bin']
    if suffix == '.pcd':
        #return np.fromfile(file_path, dtype=np.float32).reshape(-1, dim)
        with open(file_path,'r') as f:
            lines=f.readlines()
        data_index=-1
        for i in range(len(lines)):
            if lines[i].startswith("DATA "):
                data_index = i+1
                break
        if data_index ==-1:
            raise "not find Data"
        p=[]
        for i in range(data_index,len(lines)):
            vs = lines[i].strip().split()
            if len(vs) != 4:
                continue
            if lines[i] == '0 0 0 0' and i < len(lines) and lines[i+1]=='0 0 0 0':
                break

        
            #--------------------
            # the y coordinate in carla needs to be changed to -y as kitti has a different coordinate system
            # ref: https://github.com/Ozzyz/carla-data-export/blob/26c0bec203a2f3d370ff8373ca6371b7eef35300/dataexport.py#L54
            vs[1] = -1*vs[1]
            #----------------------
            p.append([float(v) for v in vs])
        return np.asarray(p, dtype=np.float32)
    elif suffix == '.bin':
        return read_points2(file_path, dim)
    else:
        raise NotImplementedError


def write_points(lidar_points, file_path):
    suffix = os.path.splitext(file_path)[1] 
    assert suffix in ['.bin', '.ply']
    if suffix == '.bin':
        with open(file_path, 'w') as f:
            lidar_points.tofile(f)
    else:
        raise NotImplementedError


def read_calib(file_path, extend_matrix=True):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    P0 = np.array([item for item in lines[0].split(' ')[1:]], dtype=np.float).reshape(3, 4)
    P1 = np.array([item for item in lines[1].split(' ')[1:]], dtype=np.float).reshape(3, 4)
    P2 = np.array([item for item in lines[2].split(' ')[1:]], dtype=np.float).reshape(3, 4)
    P3 = np.array([item for item in lines[3].split(' ')[1:]], dtype=np.float).reshape(3, 4)

    R0_rect = np.array([item for item in lines[4].split(' ')[1:]], dtype=np.float).reshape(3, 3)
    Tr_velo_to_cam = np.array([item for item in lines[5].split(' ')[1:]], dtype=np.float).reshape(3, 4)
    Tr_imu_to_velo = np.array([item for item in lines[6].split(' ')[1:]], dtype=np.float).reshape(3, 4)

    if extend_matrix:
        P0 = np.concatenate([P0, np.array([[0, 0, 0, 1]])], axis=0)
        P1 = np.concatenate([P1, np.array([[0, 0, 0, 1]])], axis=0)
        P2 = np.concatenate([P2, np.array([[0, 0, 0, 1]])], axis=0)
        P3 = np.concatenate([P3, np.array([[0, 0, 0, 1]])], axis=0)

        R0_rect_extend = np.eye(4, dtype=R0_rect.dtype)
        R0_rect_extend[:3, :3] = R0_rect
        R0_rect = R0_rect_extend

        Tr_velo_to_cam = np.concatenate([Tr_velo_to_cam, np.array([[0, 0, 0, 1]])], axis=0)
        Tr_imu_to_velo = np.concatenate([Tr_imu_to_velo, np.array([[0, 0, 0, 1]])], axis=0)

    calib_dict=dict(
        P0=P0,
        P1=P1,
        P2=P2,
        P3=P3,
        R0_rect=R0_rect,
        Tr_velo_to_cam=Tr_velo_to_cam,
        Tr_imu_to_velo=Tr_imu_to_velo
    )
    return calib_dict


# Dajiang changed
def read_label(file_path):
    with open(file_path, 'r') as f:
        data=yaml.safe_load(f)
    annotation = {}
    annotation['P0']=data['lidar0_pose']
    annotation['P1']=data['lidar1_pose']
    annotation['P2']=data['lidar2_pose']
    annotation['P3']=data['lidar3_pose']

    # everything will be processed under the coordinate system of Lidar0
    # therefore, we use the the pose of lidar0 as the reference
    lidar_pose_ref = annotation['P0']
    
    name=[]
    bbox=[]
    dimension=[]
    location=[]
    ry=[]
    for k,v in data['vehicles'].items():
        # # Dajiang changed: v['angle'][1]/180 => (v['angle'][1])*math.pi/180
        ry.append((v['angle'][1])*math.pi/180)
        
        # Dajiang changed: 'center' => location
        #location.append(v['center'])
        # transform location of the vehicle bounding box from carla world coordinate to LiDAR coordinate
        location_in_carlaworld = v['location']
        location_in_carlalidar = location_carlaworld2lidar(location_in_carlaworld,lidar_pose_ref)
        #--------------------
        # the y coordinate in carla needs to be changed to -y as kitti has a different coordinate system
        # ref: https://github.com/Ozzyz/carla-data-export/blob/26c0bec203a2f3d370ff8373ca6371b7eef35300/dataexport.py#L54
        location_in_carlalidar[1] = -1*location_in_carlalidar[1]
        location.append(location_in_carlalidar)
        extent_in_carla = v['extent']
        # refer to carla document to understand what extent.x/y/z in bounding box means
        # https://carla.readthedocs.io/en/0.9.5/measurements/
        height_kitti = 2*extent_in_carla[2] # extent.z in carla. 
        width_kitti = 2*extent_in_carla[1] # extent.y in carla
        length_kitti = 2*extent_in_carla[0] # extent.x in carla
        dimension.append(v['extent'])
        bbox.append(v['location'])
        name.append('Car')
    annotation['name'] = np.asarray(name)
    annotation['bbox'] = np.asarray(bbox, dtype=np.float)
    annotation['dimensions'] = np.asarray(dimension, dtype=np.float)
    annotation['location'] = np.asarray(location, dtype=np.float)
    annotation['rotation_y'] = np.asarray(ry, dtype=np.float)    
    return annotation

def read_label1(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    lines = [line.strip().split(' ') for line in lines]
    annotation = {}
    annotation['name'] = np.array([line[0] for line in lines])
    annotation['truncated'] = np.array([line[1] for line in lines], dtype=np.float)
    annotation['occluded'] = np.array([line[2] for line in lines], dtype=np.int)
    annotation['alpha'] = np.array([line[3] for line in lines], dtype=np.float)
    annotation['bbox'] = np.array([line[4:8] for line in lines], dtype=np.float)
    annotation['dimensions'] = np.array([line[8:11] for line in lines], dtype=np.float)[:, [2, 0, 1]] # hwl -> camera coordinates (lhw)
    annotation['location'] = np.array([line[11:14] for line in lines], dtype=np.float)
    annotation['rotation_y'] = np.array([line[14] for line in lines], dtype=np.float)
    
    return annotation


def write_label(result, file_path, suffix='.txt'):
    '''
    result: dict,
    file_path: str
    '''
    assert os.path.splitext(file_path)[1] == suffix
    name, truncated, occluded, alpha, bbox, dimensions, location, rotation_y, score = \
        result['name'], result['truncated'], result['occluded'], result['alpha'], \
        result['bbox'], result['dimensions'], result['location'], result['rotation_y'], \
        result['score']
    
    with open(file_path, 'w') as f:
        for i in range(len(name)):
            bbox_str = ' '.join(map(str, bbox[i]))
            hwl = ' '.join(map(str, dimensions[i]))
            xyz = ' '.join(map(str, location[i]))
            line = f'{name[i]} {truncated[i]} {occluded[i]} {alpha[i]} {bbox_str} {hwl} {xyz} {rotation_y[i]} {score[i]}\n'
            f.writelines(line)
