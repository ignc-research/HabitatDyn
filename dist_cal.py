import json
import logging
import os

import cv2
import argparse
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance as sci_dis
import torch
import yaml
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from tqdm import tqdm

import utils.distance_estimation as dis
from config.default import Config as CN
from utils.common import intersect2d, safe_mkdir, union2d
from utils.create_training_data import (get_pixel_number_from_name,
                                        gt_obj_points)

parser = argparse.ArgumentParser(
    description='extract pose/location info of top-down view using dectection results')
# TODO add config.yaml argument
parser.add_argument('--exp_name', default=None,
                    help='save results to dist_eval_results/exp_name')
parser.add_argument('--habitatDyn_data', metavar='DIR',
                    help='path to habitatDyn dataset', required=True)
parser.add_argument('--mask_data', metavar='DIR',
                    help='path to moving object detection output mask', required=True)
args = parser.parse_args()

# set up the parameters
USE_GT = False
SHOW_IMG = False
ADD_TRAIN_GT = True
match_rate_th = 0

# open and load config file
with open("config/test.yaml", "r") as yamlfile:
    config_yaml = yaml.load(yamlfile, Loader=yaml.FullLoader)
    print("Read successful")
print(config_yaml)

config = CN(config_yaml)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

map_size = config.SEMANTIC_ANTICIPATOR.map_size
map_scale = config.SEMANTIC_ANTICIPATOR.map_scale

# change the setting according to the dataset
# camera height of 1.25m has a pitch value pi/8, 0.20m has a pitch value 0
projection = dis.GTEgoMap(
    map_size=map_size, map_scale=map_scale, camera_pitch=np.pi/8)

imgh = config.SEMANTIC_ANTICIPATOR.imgh
imgw = config.SEMANTIC_ANTICIPATOR.imgw
pad = 0.1

# font settings
# font type
font = cv2.FONT_HERSHEY_SIMPLEX
# fontScale
fontScale = 0.5
# Blue color in BGR
color = (255, 0, 0)
# Line thickness of 2 px
thickness = 1
start_clip = 0

x_matrix = (np.arange(map_size) - np.int32(map_size/2) -
            1)[np.newaxis]*map_scale*(np.ones(map_size)[:, np.newaxis])
x_tensor = torch.tensor(x_matrix[np.newaxis, np.newaxis]).to(device)
z_matrix = -(map_size - np.arange(map_size) -
             1)[:, np.newaxis]*np.ones(map_size)[np.newaxis, :]*map_scale
z_tensor = torch.tensor(z_matrix[np.newaxis, np.newaxis]).to(device)

img_dataset_list = [args.habitatDyn_data]
pred_list = [args.mask_data]

if not args.exp_name:
    dir_count = sum(os.path.isdir(os.path.join('./dist_eval_results/', i))
                    for i in os.listdir(f'./dist_eval_results'))
    exp_name = f"exp_{dir_count:03d}"
else:
    exp_name = args.exp_name

for i in range(len(img_dataset_list)):
    # the habitatDyn sub dataset to be evluated on distance estimation
    dataset_name = img_dataset_list[i].split("/")[-1]
    data_root = img_dataset_list[i]
    root_file = os.path.join(data_root, 'habitat_sim_DAVIS/Annotations/')
    img_path = os.path.join(data_root, 'habitat_sim_DAVIS/JPEGImages/480p')
    clips_name = os.listdir(root_file + '480p')
    clips_name = sorted(clips_name)
    frame_list = os.listdir(root_file + '480p/0000')
    frame_list = sorted(frame_list)

    for clip in tqdm(clips_name[start_clip:], desc="Clip", position=0):
        meta_file_location = os.path.join(data_root, f'stats_info/480p/{clip}')
        with open(os.path.join(meta_file_location, 'semantic_id_to_name.json')) as f:
            id_to_name = json.load(f)

        im_folder = ''
        focal = ''
        safe_mkdir('dist_eval_results')

        if USE_GT:
            record_save_path = f'dist_eval_results/{exp_name}/{dataset_name}/use_gt/{focal}{im_folder}/'
        else:
            record_save_path = f'dist_eval_results/{exp_name}/{dataset_name}/use_pre/{focal}{im_folder}/'

        safe_mkdir(record_save_path)

        logging.basicConfig(level=logging.DEBUG, filename=record_save_path + "logfile", filemode="a+",
                            format="%(asctime)-15s %(levelname)-8s %(message)s")
        eval_record = []
        error = []
        right_pixel = 0
        wrong_pixel = 0
        wrong_object = 0
        timecost = 0
        eval_time = 0
        for frame in tqdm(frame_list, desc="Frame", position=1, leave=False):
            fram_num = int(frame.split('.')[0])
            pose_camera = np.load(
                os.path.join(meta_file_location, 'camera_spec.npy'), allow_pickle=True).tolist()
            pose_ped = np.load(os.path.join(meta_file_location,
                               'peds_infos.npy'), allow_pickle=True).tolist()
            location_camera = pose_camera['position']
            location_ped = pose_ped['positions']
            ground_truth = np.uint8(cv2.imread(
                root_file + '480p_objectID/' + clip + '/' + frame)[:, :, 0])
            depth_img = cv2.imread(
                root_file + '480p_depth/' + clip + '/' + frame)
            depth_array = depth_img[:, :, 0]/255.
            ground_truth_points = {}
            gt_mask = ground_truth > 0
            object_ids = np.unique(ground_truth)

            # validation object:
            valid_object_ids = []
            for object_id in object_ids[1:]:
                location = location_ped[object_id-1][fram_num]
                dis_gt_2 = location[0]*location[0] + location[2]*location[2]
                dis_gt = np.sqrt(dis_gt_2)
                object_mask = ground_truth == object_id
                object_name = id_to_name[str(object_id)]
                pixel_th = get_pixel_number_from_name(object_name)
                if dis_gt > 1:
                    beta = 0.5
                if dis_gt < 1:
                    beta = 0.2

                if np.sum(object_mask)*dis_gt_2 > beta*pixel_th:
                    valid_object_ids.append(object_id)

            if USE_GT:
                # mask_img = cv2.imread(
                #     root_file + '480p_colored/' + clip + '/' + frame)
                # mask_img_1d = np.sum(mask_img, axis=2)
                # mask = mask_img_1d > 0
                raise(NotImplementedError)
            else:
                mask_file_path = args.mask_data
                mask_img = cv2.imread(os.path.join(
                    mask_file_path, f'{clip}/{frame[1:]}'))
                mask = np.sum(mask_img, axis=2) > 0

            # if all the objects are not detected
            if np.all(mask == False):
                if not np.all(gt_mask == False):
                    for object_id in valid_object_ids:
                        obj_points, ref_position = gt_obj_points(pose_ped, pose_camera,
                                                                 fram_num, object_id, map_size, map_scale, id_to_name)
                        record = {
                            'clip': clip,
                            'frame': fram_num,
                            'object_id': object_id,
                            'obj': id_to_name[str(object_id)],
                            'detected': False,
                            'pre_location': None,
                            'gt_location': ref_position,
                            'intersect': 0,
                            'union': len(obj_points)
                        }
                        #print(f'!!!object detect failed, record:{record} \n')
                        eval_record.append(record)
                continue

            ego_map_masked = projection.get_observation(depth_array*mask)
            # TODO: why?
            dilation_kernel = np.ones((5, 5))
            dilation_mask = cv2.dilate(ego_map_masked[:, :, 0], dilation_kernel, iterations=2,
                                       ).astype(np.float32)
            points = np.array(np.where(ego_map_masked[:, :, 0] > 0.3)).T

            # TODO: why duplicate
            if points.shape[0] < 2:
                # if all the objects are not detected
                if not np.all(gt_mask == False):
                    for object_id in valid_object_ids:
                        obj_points, ref_position = gt_obj_points(pose_ped, pose_camera,
                                                                 fram_num, object_id, map_size, map_scale, id_to_name)
                        record = {
                            'clip': clip,
                            'frame': fram_num,
                            'object_id': object_id,
                            'obj': id_to_name[str(object_id)],
                            'detected': False,
                            'pre_location': None,
                            'gt_location': ref_position,
                            'intersect': 0,
                            'union': len(obj_points)
                        }
                        #print(f'!!!object detect failed, record:{record} \n')
                        eval_record.append(record)
                continue

            clf1 = LocalOutlierFactor(n_neighbors=np.min(
                [int(points.shape[0]/5+1), 50]), contamination=0.3)
            y_pred = clf1.fit_predict(points)
            if SHOW_IMG:
                plt.figure(figsize=(10, 10))
                plt.xlim([0, map_size])
                plt.ylim([0, map_size])
                plt.axis('off')
                plt.scatter(points[:, 1], points[:, 0], color="k", s=1.0)
                plt.scatter(points[y_pred > 0, 1],
                            points[y_pred > 0, 0], color="r", s=1.0)
                plt.show()
            clustering = DBSCAN(eps=3, min_samples=2).fit(points[y_pred > 0])
            relativ_average_cors = {}
            valid_clusters = []
            # matching cluster with ground truth labels in top-down view
            # TODO: clear all "_o" variable
            for lable in np.unique(clustering.labels_):
                o_cluster_points = points[y_pred > 0][np.where(
                    clustering.labels_ == lable)]
                other_cluster_points = points[y_pred > 0][np.where(
                    clustering.labels_ != lable)]
                if o_cluster_points.shape[0] > 2:
                    valid_clusters.append(lable)
                    shift_centre = []
                    cluster_points = points[y_pred > 0][np.where(
                        clustering.labels_ == lable)] - [map_size, int(map_size/2)+1]
                    cluster_points = cluster_points*map_scale
                    cluster_centre = np.average(cluster_points, axis=0)
                    x_pt_o = cluster_centre[1]
                    z_pt_o = cluster_centre[0]
                    cluster_score = 0
                    cluster_score_o = 0

                    x_pt = x_pt_o
                    z_pt = z_pt_o
                    cluster_ext = o_cluster_points
                    ext_points_img = None
                    imagine_scores = np.zeros((4, 9))

                    cluster_points[:, 1] = - cluster_points[:, 1]
                    cluster_points = cluster_points*map_scale
                    cluster_centre = np.average(cluster_points, axis=0)
                    relativ_average_cors[lable] = {
                        'orignal_points': cluster_ext,
                        'cluster_points': cluster_points,
                        'cluster_centre': [x_pt, z_pt],
                        'cluster_centre_o': [x_pt_o, z_pt_o],
                        'cluster_centre_no_shift': shift_centre,
                        'cluster_score': cluster_score,
                        'cluster_score_o': cluster_score_o,
                        'cluster_o': o_cluster_points,
                        'imagination_score': imagine_scores,
                    }

            # calculate ground truth points for each label and stored in ground_truth_points dict
            for object_id in object_ids[1:]:
                mask = ground_truth == object_id
                ego_map_gt_object = projection.get_observation(
                    depth_array*mask)
                points = np.array(np.where(ego_map_gt_object[:, :, 0] > 0)).T
                obj_points, ref_position = gt_obj_points(pose_ped, pose_camera,
                                                         fram_num, object_id, map_size, map_scale, id_to_name)
                if points.shape[0] > 0:
                    clf = LocalOutlierFactor(n_neighbors=np.min(
                        [int(points.shape[0]/5+1), 50]), contamination=0.2)
                    y_pred = clf.fit_predict(points)
                    location = pose_ped['positions'][object_id-1][fram_num]
                    rotation = pose_camera['agent_orient'][fram_num]
                    obj_heading = pose_ped['orientations'][object_id-1][fram_num]
                    ground_truth_points[object_id] = {
                        'mask_points': points[y_pred > 0],
                        'gt_points': np.array(obj_points),
                        'ref_position': ref_position,
                        'location': location,
                        'rob_rotation': rotation,
                        'obj_heading': obj_heading
                    }

            # find the match between estimated data and ground truth data
            # here the 'label' is the label of DBSCAN cluster, so check whether is just numbering or meaningful label for habitatDyn
            id2lable = {}
            for object_id in valid_object_ids:
                if object_id in ground_truth_points.keys():
                    best_match = {
                        'lable': -1,
                        'points_match': 0
                    }  # -1 no match, 0 pints match
                    for lable in valid_clusters:
                        points_pre = relativ_average_cors[lable]['orignal_points']

                        points_gt = union2d(
                            ground_truth_points[object_id]['mask_points'], ground_truth_points[object_id]['gt_points'])

                        points_matched = intersect2d(points_pre, points_gt)
                        match_rate = points_matched.shape[0]/points_gt.shape[0]
                        if points_matched.shape[0] > best_match['points_match'] and match_rate > match_rate_th:
                            best_match['lable'] = lable
                            best_match['points_match'] = points_matched.shape[0]
                    id2lable[object_id] = best_match['lable']

            # rename
            rocorded_pre_label = []

            for object_id in id2lable.keys():
                # if no cluster for certain gt object_id found
                if id2lable[object_id] == -1:
                    record = {
                        'clip': clip,
                        'frame': fram_num,
                        'object_id': object_id,
                        'obj': id_to_name[str(object_id)],
                        'detected': False,
                        'pre_location': None,
                        'gt_location': ground_truth_points[object_id]['ref_position'],
                        'intersect': 0,
                        'union': len(ground_truth_points[object_id]['gt_points']),
                        'location': ground_truth_points[object_id]['location'],
                        'rotation': ground_truth_points[object_id]['rob_rotation'],
                        'obj_heading': ground_truth_points[object_id]['obj_heading']
                    }
                    # print(record)
                    eval_record.append(record)
                else:
                    points_pre = relativ_average_cors[id2lable[object_id]
                                                      ]['orignal_points']
                    points_pre_o = relativ_average_cors[id2lable[object_id]]['cluster_o']
                    points_gt = ground_truth_points[object_id]['gt_points']

                    points_match_o = intersect2d(points_pre_o, points_gt)

                    points_matched = intersect2d(points_pre, points_gt)

                    x = relativ_average_cors[id2lable[object_id]
                                             ]['cluster_centre'][0]
                    z = relativ_average_cors[id2lable[object_id]
                                             ]['cluster_centre'][1]

                    x_object_map_coor = x/map_scale + int(map_size/2) + 1
                    z_object_map_coor = z/map_scale + map_size

                    distance_ct = sci_dis.cdist(points_pre, np.array(
                        [z_object_map_coor, x_object_map_coor])[np.newaxis])
                    r = max(distance_ct)

                    record = {
                        'clip': clip,
                        'frame': fram_num,
                        'object_id': object_id,
                        'lable': lable,
                        'obj': id_to_name[str(object_id)],
                        'score': relativ_average_cors[lable]['cluster_score'],
                        'score_o': relativ_average_cors[lable]['cluster_score_o'],
                        'detected': True,
                        'pre_location': [x, z],
                        'pre_location_o': relativ_average_cors[lable]['cluster_centre_o'],
                        'no_shift_location': relativ_average_cors[id2lable[object_id]]['cluster_centre_no_shift'],
                        'gt_location': ground_truth_points[object_id]['ref_position'],
                        'intersect': points_matched.shape[0],
                        'intersect_o': len(points_match_o),
                        'union': len(points_gt),
                        'detected_size': len(points_pre),
                        'detected_size_0': len(points_pre_o),
                        'location': ground_truth_points[object_id]['location'],
                        'rotation': ground_truth_points[object_id]['rob_rotation'],
                        'obj_heading': ground_truth_points[object_id]['obj_heading'],
                        'r': r,
                        'imagination_score': relativ_average_cors[id2lable[object_id]]['imagination_score'],
                    }
                    # print(record)
                    right_pixel += points_matched.shape[0]
                    eval_record.append(record)
                    error.append(
                        np.array(record['pre_location']) - np.array(record['gt_location']))
                    rocorded_pre_label.append(id2lable[object_id])

            # if a predicted cluster not matched(e.g moving object segmentation model predicted 2 cluster for same moving object_id, the smaller cluster will be ignored by before ops)
            for lable in valid_clusters:
                if lable not in rocorded_pre_label:
                    points_pre = relativ_average_cors[lable]['orignal_points']
                    points_pre_o = relativ_average_cors[lable]['cluster_o']
                    best_match = {
                        'object_id': -1,
                        'points_match': 0
                    }  # -1 no match, 0 pints match
                    for object_id in ground_truth_points.keys():
                        points_gt = union2d(
                            ground_truth_points[object_id]['mask_points'], ground_truth_points[object_id]['gt_points'])
                        points_matched = intersect2d(points_pre, points_gt)
                        if points_matched.shape[0] > best_match['points_match']:
                            best_match['object_id'] = lable
                            best_match['points_match'] = points_matched.shape[0]
                    alpha = best_match['points_match']/len(points_pre)
                    x = relativ_average_cors[lable]['cluster_centre'][0]
                    z = relativ_average_cors[lable]['cluster_centre'][1]

                    x_object_map_coor = x/map_scale + int(map_size/2) + 1
                    z_object_map_coor = z/map_scale + map_size

                    distance_ct = sci_dis.cdist(points_pre, np.array(
                        [z_object_map_coor, x_object_map_coor])[np.newaxis])
                    r = max(distance_ct)

                    # TODO: why only match a small amount, why gt is [0, 0]
                    if alpha < 0.1:
                        record = {
                            'clip': clip,
                            'frame': fram_num,
                            'lable': lable,
                            'obj': 'unknown',
                            'score': relativ_average_cors[lable]['cluster_score'],
                            'score_o': relativ_average_cors[lable]['cluster_score_o'],
                            'detected': True,
                            'pre_location': [x, z],
                            'gt_location': [0, 0],
                            'intersect':  best_match['points_match'],
                            'union': len(points_pre),
                            'detected_size': len(points_pre),
                            'detected_size_0': len(points_pre_o),
                            'gt_ob_id': valid_object_ids,
                            'r': r,
                            'imagination_score': relativ_average_cors[lable]['imagination_score'],
                        }
                        if len(valid_object_ids) > 0:
                            wrong_pixel += len(points_pre)
                            wrong_object += 1
                        eval_record.append(record)

        if len(error) > 0:
            mse = np.mean(np.array(error)[:, 0] ** 2) + \
                np.mean(np.array(error)[:, 1] ** 2)
            print(record_save_path)
            print(f'clip:{clip} error: {mse} detected: {len(error)} \
                \n pixel for obejct : {right_pixel}, wrong pixel: {wrong_pixel} \
                \n wrong_object: {wrong_object} timecost: {timecost/(eval_time+0.001)}')
            logging.info(f'clip:{clip} error: {mse} detected: {len(error)} \
                \n pixel for obejct : {right_pixel}, wrong pixel: {wrong_pixel} \
                \n wrong_object: {wrong_object} timecost: {timecost/(eval_time+0.001)}')

        np.save(record_save_path + clip + '.npy', eval_record)
