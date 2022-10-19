import numpy as np
import os
import math

def x_to_world(pose):
    """
    The transformation matrix from x-coordinate system to carla world system
    Parameters
    ----------
    pose : list
        [x, y, z, roll, yaw, pitch]
    Returns
    -------
    matrix : np.ndarray
        The transformation matrix.
    """
    x, y, z, roll, yaw, pitch = pose[:]

    # used for rotation matrix
    c_y = np.cos(np.radians(yaw))
    s_y = np.sin(np.radians(yaw))
    c_r = np.cos(np.radians(roll))
    s_r = np.sin(np.radians(roll))
    c_p = np.cos(np.radians(pitch))
    s_p = np.sin(np.radians(pitch))

    matrix = np.identity(4)
    # translation matrix
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z

    # rotation matrix
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r

    return matrix

def location_carlaworld2lidar(location_in_carlaworld,lidar_pose):
    """
    project locations in carld world coordinates to lidar coordinate
    Note lidar can be substitute by the pose of any objects under carla world 

    """
    lidar_to_world_transform = x_to_world(lidar_pose)
    world_to_lidar_transform = np.linalg.inv(lidar_to_world_transform)

    location_in_carlaworld = location_in_carlaworld +[1]
    location_in_carlaworld = np.array(location_in_carlaworld)
    location_in_carlaworld = np.expand_dims(location_in_carlaworld, axis=0).T

    # note we will get an (4,1) array while the first three elements x/y/z is what we need
    # and we need to transform the array into a list
    location_in_lidar_array = np.dot(world_to_lidar_transform,location_in_carlaworld)
    location_in_lidar_array = location_in_lidar_array.squeeze()

    location_in_lidar = [location_in_lidar_array[0],location_in_lidar_array[1],location_in_lidar_array[2]]

    return location_in_lidar



	

def location_carlalidar2kitti(bbox_world_carla,lidar_pose):
	"""
        derive gt bounding boxes in kitti format (x,y,z,h,w,l,yaw) 
        based on gt bounding boxes in carla format. 
        Parameters
        ----------
        bbox_world_carla : list
            List of dictionary, save all cavs' information.
        lidar_pose : list
            The final target lidar pose with length 6.
        Returns
        -------
        bbox_lidar_kitti : np.ndarray
            Shape is (max_num, 7).
        mask : np.ndarray
            Shape is (max_num,).
        object_ids : list
            Length is number of bbx in current sample.
    """

def pointcloud_carlalidartokittilidar(pc_carla,pc_kitti):

    # ref: https://github.com/Ozzyz/carla-data-export/blob/26c0bec203a2f3d370ff8373ca6371b7eef35300/dataexport.py#L54
    """ Saves lidar data to given filename, according to the lidar data format.
        bin is used for KITTI-data format, while .ply is the regular point cloud format
        In Unreal, the coordinate system of the engine is defined as, which is the same as the lidar points
        z
        ^   ^ x
        |  /
        | /
        |/____> y
        This is a left-handed coordinate system, with x being forward, y to the right and z up
        See also https://github.com/carla-simulator/carla/issues/498
        However, the lidar coordinate system from KITTI is defined as
              z
              ^   ^ x
              |  /
              | /
        y<____|/
        Which is a right handed coordinate sylstem
        Therefore, we need to flip the y axis of the lidar in order to get the correct lidar format for kitti.
        This corresponds to the following changes from Carla to Kitti
            Carla: X   Y   Z
            KITTI: X  -Y   Z
        NOTE: We do not flip the coordinate system when saving to .ply.
    """
    return


    