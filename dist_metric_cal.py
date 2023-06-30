import argparse
import logging
import math
import os

import numpy as np

from utils.common import safe_mkdir

# TODO add sub-drectory for each exp
# TODO argparser for detect ranger
parser = argparse.ArgumentParser(
    description='Calculate metrics for distance estimation results')
parser.add_argument('--data_path', metavar='DIR',
                    help='path to parsed .npy files', required=True)
parser.add_argument('--range_start', type=int,
                    help='starting distance for evalutation', required=True)
parser.add_argument('--range_end', type=int,
                    help='end distance for evaluation', required=True)
args = parser.parse_args()

safe_mkdir('./dist_metric_results')
logging.basicConfig(level=logging.DEBUG, filename=os.path.join('./dist_metric_results', "logfile"), filemode="a+",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")

data_path = args.data_path
file_list = os.listdir(data_path)


# distance in top-down view
def dist_cal(x, y):
    return math.sqrt(x**2 + y**2)


print('loading...')
# lambda function used to map the shape of element of record_all: (3,2):[pre_location pair, gt_location pair, (pre_dist, gt_dist) pair]


def read_dist(x): return np.array([x['pre_location'], x['gt_location'],
                                   [dist_cal(*x['pre_location']),
                                    dist_cal(*x['gt_location'])]]) if x['detected'] else np.array([[None, None], x['gt_location'], [None, dist_cal(*x['gt_location'])]])


record_all = np.array([])
for record_name in file_list:
    if not record_name.endswith(".npy"):
        continue
    try:
        record = np.load(os.path.join(data_path,record_name), allow_pickle=True)
    except:
        print(f".npy file {record_name} loading failed, continue")
        continue

    if record_all.shape[0] == 0:
        record_all = np.array(list(map(read_dist, record)))
    elif len(record) != 0:
        record_all = np.concatenate(
            (record_all, np.array(list(map(read_dist, record)))), axis=0)

print("prepare data...")
record_all_tmp = record_all
index = np.where((record_all_tmp[:, 2, 1] > args.range_start))
record_all_tmp = record_all_tmp[index]
index = np.where((record_all_tmp[:, 2, 1] <= args.range_end))
record_all_tmp = record_all_tmp[index]
# detction chance
successful_detection_times = np.sum(
    np.where(record_all_tmp[:, 0, 0] != None, 1, 0))
successful_rate = successful_detection_times/record_all_tmp.shape[0]
print(f'successful_rate: {successful_rate} ')

# Threshold
# index = record_all[:,0] == 1
index = record_all_tmp[:, 0, 0] != None
# detected_frame = record_all[index]
detected_frame = record_all_tmp[index]
# distance to close will be ignored
detected_frame = detected_frame[detected_frame[:, 2, 1] > 0.01]
theshold_o = 1.25
est = np.array(detected_frame[:, 2, 0], dtype=np.float32)
gt = np.array(detected_frame[:, 2, 1], dtype=np.float32)
est_over_gt = est / gt
gt_over_est = gt / est
for i in range(3):
    theshold = theshold_o**(i+1)
    index = ((est_over_gt < theshold) * (gt_over_est < theshold)) > 0
    print(
        f'threshold: {i+1}, percentage: {np.sum(index)/detected_frame.shape[0]}')


mae = np.mean(np.abs(est - gt))
mae_res = f"Mean Absolute Error: {mae}"
print(mae_res)
logging.info(mae_res)

squared_relative_error = ((est - gt) / gt) ** 2
mean_squared_relative_error = np.mean(squared_relative_error)
msre_res = f"Mean Squared Relative Error: {mean_squared_relative_error}",
print(msre_res)
logging.info(msre_res)

# Calculate the root mean squared error
mse = np.mean((est - gt) ** 2)
rmse = np.sqrt(mse)
rmse_res = f"Root Mean Squared Error: {rmse}"
print(rmse_res)
logging.info(rmse_res)

# Calculate the root mean squared logarithmic error
squared_log_error = (np.log1p(est) - np.log1p(gt)) ** 2
msle = np.mean(squared_log_error)
rmsle = np.sqrt(msle)
rmsle_res = f"Root Mean Squared Logarithmic Error: {rmsle}"
print(rmsle_res)
logging.info(rmsle_res)

# Calculate the root mean squared error of location (x,y)
filtered_detected_frame = np.array(
    [x[:2] for x in detected_frame if x[0, 0] != None])
x_sqr = (filtered_detected_frame[:, 0, 0] -
         filtered_detected_frame[:, 1, 0])**2
y_sqr = (filtered_detected_frame[:, 0, 1] -
         filtered_detected_frame[:, 1, 1])**2
xy = np.concatenate((x_sqr, y_sqr), axis=0)
rmse_xy = np.sqrt(np.mean(xy))
rmse_xy_res = f"Root Mean squared Error for location (x,y) is {rmse_xy}"
print(rmse_xy_res)
logging.info(rmse_xy_res)
