import math
import numpy as np
from scipy.spatial import ConvexHull
from utils.geometry_utils import (
        quaternion_from_coeff,
        compute_heading_from_quaternion,quaternion_xyzw_to_wxyz, 
        compute_quaternion_from_heading,
        quaternion_rotate_vector
        )

def rectangle_coordinates(center:list, width, length, orientation):
    '''
    center[0]: int x, real_point_col
    center[1]: int y, real_point_row
    width: int the width of the object
    length: int the length of the object
    orientation: float angle in RAD, counterclockwise
    return is the pixel corner of the rectangle
    '''
    # Calculate the width and length offsets
    half_width = width / 2
    half_length = length / 2

    # Calculate the coordinates of the rectangle's corners relative to the center
    dx1 = half_width * math.cos(orientation) - half_length * math.sin(orientation)
    dy1 = half_width * math.sin(orientation) + half_length * math.cos(orientation)

    dx2 = half_width * math.cos(orientation) + half_length * math.sin(orientation)
    dy2 = half_width * math.sin(orientation) - half_length * math.cos(orientation)

    # Calculate the pixel locations of the corners of the rectangle
    p1 = [int(center[0] + dx1), int(center[1] - dy1)]
    p2 = [int(center[0] + dx2), int(center[1] - dy2)]
    p3 = [int(center[0] - dx1), int(center[1] + dy1)]
    p4 = [int(center[0] - dx2), int(center[1] + dy2)]

    return [p1, p2, p3, p4]


def in_rectangle(p, hull):
    new_points = np.vstack((hull.points, p))
    new_hull = ConvexHull(new_points)
    return list(hull.vertices) == list(new_hull.vertices)

def find_points_inside_rectangle(xv,yv,rect_points):
    '''
    give the corners of the rectangle and a list of the points,
    the out put is a list for points in the rectangle.
    '''
    inside_points = []
    hull = ConvexHull(rect_points)
    for x in xv:
        for y in yv:
            if in_rectangle((x, y), hull):
                # in a image y is col and x is line
                inside_points.append([y, x])

    return inside_points

def get_size_from_name(name,pad=0):
    width = 0
    height = 0
    if name == 'angry_girl':
        width, height = [0.48, 0.32]
    if name == 'robot_2020':
        width, height = [0.68, 0.42]
    if name == 'miniature_cat':
        width, height = [0.12, 0.45]
    if name == 'ferbibliotecario':
        width, height = [0.53, 0.245]
    if name == 'shiba':
        width, height = [0.164, 0.34]
    if name == 'toy_car':
        width, height = [0.12, 0.32]
    if name == 'big_bot':
        width, height = [1.02, 0.68]
    if name == 'noirnwa':
        width, height = [0.74, 0.32]
    if name == 'tsai_ing-wen':
        width, height = [0.74, 0.28]           
    return width + pad, height + pad

def gt_obj_points(pose_ped, pose_camera, fram_num, object_id, map_size, map_scale, id_to_name, pad=0):
    '''
    pose_ped: the dict read from the file peds_infos.npy
    pose_camera: the dict read from the file camera_spec.npy
    fram_num: the frame number int
    object_id: the object id int
    '''
    location = pose_ped['positions'][object_id-1][fram_num] # - pose_camera['position'][fram_num]
    rotation = quaternion_from_coeff(pose_camera['agent_orient'][fram_num])
    agent_heading = compute_heading_from_quaternion(rotation)
    agent_heading_qua = compute_quaternion_from_heading(agent_heading)
    location_ref = quaternion_rotate_vector(agent_heading_qua.inverse(),location)
    real_point_col = np.round(location_ref[0]/map_scale) + int(map_size/2) + 1
    real_point_row = np.round(location_ref[2]/map_scale) + map_size
    ped_ori_qua = quaternion_xyzw_to_wxyz(pose_ped['orientations'][object_id-1][fram_num])
    ped_ori_heading = -(compute_heading_from_quaternion(ped_ori_qua) - agent_heading)
    
    real_point_col = min(max(0,real_point_col),map_size-1)
    real_point_row = min(max(0,real_point_row),map_size-1)

    object_name = id_to_name[str(object_id)]
    width, height = get_size_from_name(object_name,pad)
    rect_points = np.array(rectangle_coordinates([real_point_col,real_point_row],int(width/map_scale),int(height/map_scale), ped_ori_heading))
    x_min = np.min(rect_points[:,0])
    x_max = np.max(rect_points[:,0])
    y_min = np.min(rect_points[:,1])
    y_max = np.max(rect_points[:,1])
    return find_points_inside_rectangle(range(x_min,x_max), range(y_min,y_max),rect_points), [location_ref[0],location_ref[2]]

def gt_ref_location(pose_ped, pose_camera, fram_num, object_id):
    '''
    pose_ped: the dict read from the file peds_infos.npy
    pose_camera: the dict read from the file camera_spec.npy
    fram_num: the frame number int
    object_id: the object id int
    '''
    location = pose_ped['positions'][object_id-1][fram_num]
    rotation = quaternion_from_coeff(pose_camera['agent_orient'][fram_num])
    agent_heading = compute_heading_from_quaternion(rotation)
    agent_heading_qua = compute_quaternion_from_heading(agent_heading)
    location_ref = quaternion_rotate_vector(agent_heading_qua.inverse(),location)
    
    return [location_ref[0],location_ref[2]]


def get_pixel_number_from_name(name):
    num = 0
    if name == 'angry_girl':
        num = 28131
    if name == 'robot_2020':
        num = 22422
    if name == 'miniature_cat':
        num = 3216
    if name == 'ferbibliotecario':
        num = 32175
    if name == 'shiba':
        num = 2559
    if name == 'toy_car':
        num = 2333
    if name == 'big_bot':
        num = 69925
    if name == 'noirnwa':
        num = 39508
    if name == 'tsai_ing-wen':
        num = 36112
    return num