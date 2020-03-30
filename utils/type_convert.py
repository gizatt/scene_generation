from pydrake.math import (RollPitchYaw, RigidTransform, RotationMatrix)
from pydrake.common.eigen_geometry import Quaternion, AngleAxis, Isometry3

import numpy as np
from typing import Dict
import scene_generation.utils.transformations as transformations

def camera2world_from_map(camera2world_map):  # type: (Dict) -> np.ndarray
    """
    Get the transformation matrix from the storage map
    See ${processed_log_path}/processed/images/pose_data.yaml for an example
    :param camera2world_map:
    :return: The (4, 4) transform matrix from camera 2 world
    """
    # The rotation part
    camera2world_quat = [1, 0, 0, 0]
    camera2world_quat[0] = camera2world_map['quaternion']['w']
    camera2world_quat[1] = camera2world_map['quaternion']['x']
    camera2world_quat[2] = camera2world_map['quaternion']['y']
    camera2world_quat[3] = camera2world_map['quaternion']['z']
    camera2world_quat = np.asarray(camera2world_quat)
    camera2world_matrix = transformations.quaternion_matrix(camera2world_quat)

    # The linear part
    camera2world_matrix[0, 3] = camera2world_map['translation']['x']
    camera2world_matrix[1, 3] = camera2world_map['translation']['y']
    camera2world_matrix[2, 3] = camera2world_map['translation']['z']
    return camera2world_matrix


def camera2world_from_teleopmap(camera2world_map):  # type: (Dict) -> np.ndarray
    """
    Get the transformation matrix from the storage map
    See ${processed_log_path}/processed/images/pose_data.yaml for an example
    :param camera2world_map:
    :return: The (4, 4) transform matrix from camera 2 world
    """
    # The rotation part
    camera2world_quat = [1, 0, 0, 0]
    camera2world_quat[0] = camera2world_map['quaternion_wxyz'][0]
    camera2world_quat[1] = camera2world_map['quaternion_wxyz'][1]
    camera2world_quat[2] = camera2world_map['quaternion_wxyz'][2]
    camera2world_quat[3] = camera2world_map['quaternion_wxyz'][3]
    camera2world_quat = np.asarray(camera2world_quat)
    camera2world_matrix = transformations.quaternion_matrix(camera2world_quat)

    # The linear part
    camera2world_matrix[0, 3] = camera2world_map['position_xyz'][0]
    camera2world_matrix[1, 3] = camera2world_map['position_xyz'][1]
    camera2world_matrix[2, 3] = camera2world_map['position_xyz'][2]
    return camera2world_matrix

def transform_from_pose_vector(pose_vector):  # type: (np.array(7)) -> np.ndarray
    """
    Get the transformation matrix from a pose vector in [qw qx qy qz x y z] order.
    :param pose_vector
    :return: The (4, 4) transform matrix
    """
    # The rotation part
    pose_vector = np.array(pose_vector)
    quat = pose_vector[:4]
    tf = transformations.quaternion_matrix(quat)
    # The linear part
    tf[:3, 3] = pose_vector[-3:]
    return tf

def rigidtransform_from_pose_vector(pose_vector):
    quat_part = pose_vector[:4]
    quat_part /= np.linalg.norm(quat_part)
    # Transform keypoints to camera frame, and then camera image coordinates.
    return RigidTransform(p=pose_vector[-3:], quaternion=Quaternion(quat_part))

def pose_vector_from_rigidtransform(tf):
    return np.concatenate([Quaternion(tf.rotation().matrix()).wxyz(), tf.translation()])

def dict_to_matrix(d):
    """
    Extract a np array from a dict with data, rows, and columns entries.
    """
    return np.array(d["data"]).reshape(d["rows"], d["cols"])


def matrix_to_dict(m):
    """
    Extract a np array from a dict with data, rows, and columns entries.
    """
    return {
        "data": m.flatten().tolist(),
        "rows": m.shape[0],
        "cols": m.shape[1]
    }
