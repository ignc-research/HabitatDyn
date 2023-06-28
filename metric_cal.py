import argparse
import logging
import os

import numpy as np
import torch
import torch.utils.data.dataloader as dataloader
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from utils.common import safe_mkdir
from utils.meter import AverageValueMeter
from utils.metrics import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(
    description='PyTorch Metrics Calculator', formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--gt_data', metavar='DIR',
                    help='path to ground truth dataset', required=True)
parser.add_argument('--pred_data', metavar='DIR',
                    help='path to predicted dataset', required=True)
parser.add_argument('--flag', type=int, required=True,
                    help='''which kind HabitatDyn data to do metric evalutation:
                    flag:   0: calculate All
                            1: calculate Single class
                            2: calculate Multi class
                            3: calculate Speed 1
                            4: calculate Speed 2
                            5: calculate Speed 3
                            6: calculate Human classes
                            7: calculate toy car/robot classes
                            8: calculate dog/cat classes''')
args = parser.parse_args()


class MetricDataset(Dataset):
    """A dataset to load gt_data and pred_data

    Args: gt_data, pred_data, the root dir of ground truth and predicition
          flag: 0: calculate all
                1: calculate single class
                2: calculate Multi class
                3: calculate speed 1
                4: calculate speed 2
                5: calculate speed 3
    """

    def __init__(self, pred_data, gt_data, flag):
        self.pred_data = pred_data
        self.gt_data = gt_data
        self.flag = flag
        self.data = []

        if flag == 0:
            self.scene_names = self.load_all()
        elif flag == 1:
            self.scene_names = self.load_single_class()
        elif flag == 2:
            self.scene_names = self.load_multi_class()
        elif flag == 3:
            self.scene_names = self.load_speed_1()
        elif flag == 4:
            self.scene_names = self.load_speed_2()
        elif flag == 5:
            self.scene_names = self.load_speed_3()
        elif flag == 6:
            self.scene_names = self.load_human_class()
        elif flag == 7:
            self.scene_names = self.load_car_robot_class()
        elif flag == 8:
            self.scene_names = self.load_dog_cat_class()

        for scene_name in self.scene_names:
            gt_folder = os.path.join(
                self.gt_data, 'habitat_sim_DAVIS/Annotations/480p', scene_name)
            pred_folder = os.path.join(self.pred_data, scene_name)

            for filename in os.listdir(gt_folder):
                if filename.endswith('.png') or filename.endswith('.jpg'):
                    gt_path = os.path.join(gt_folder, filename)
                    pred_path = os.path.join(pred_folder, filename[1:])
                    if not os.path.isfile(pred_path):
                        continue
                    self.data.append((gt_path, pred_path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        pred_img = Image.open(self.data[index][1]).convert('L')
        binary_pred_img = pred_img.point(lambda x: 0 if x == 0 else 1)
        gt_img = Image.open(self.data[index][0]).convert('L')
        binary_gt_img = gt_img.point(lambda x: 0 if x == 0 else 1)

        return torch.from_numpy(np.array(binary_pred_img)), torch.from_numpy(np.array(binary_gt_img))

    def load_all(self):
        # load all data path stored in .txt file in gt_data
        txt_file = os.path.join(self.gt_data, 'video_name_mapping.txt')
        data = np.genfromtxt(txt_file, delimiter='\t', dtype='str')
        data_filtered = []
        for i in range(len(data)):
            data_filtered.append(data[i][0])

        return data_filtered

    def load_single_class(self):
        # load all data path that has single class stored in .txt file in gt_data
        # meaning first 36 entries of 54
        txt_file = os.path.join(self.gt_data, 'video_name_mapping.txt')
        data = np.genfromtxt(txt_file, delimiter='\t', dtype='str')
        data_filtered = []
        for i in range(len(data)):
            if int(data[i][0]) % 54 < 36:
                data_filtered.append(data[i][0])

        return data_filtered

    def load_multi_class(self):
        # load all data path that has multi class stored in .txt file in gt_data
        # meaning last 18 entries of 54
        txt_file = os.path.join(self.gt_data, 'video_name_mapping.txt')
        data = np.genfromtxt(txt_file, delimiter='\t', dtype='str')
        data_filtered = []
        for i in range(len(data)):
            if int(data[i][0]) % 54 >= 36:
                data_filtered.append(data[i][0])

        return data_filtered

    def load_speed_1(self):
        txt_file = os.path.join(self.gt_data, 'video_name_mapping.txt')
        data = np.genfromtxt(txt_file, delimiter='\t', dtype='str')
        data_filtered = []
        for i in range(len(data)):
            if data[i][1].split('_')[-1] == '1':
                data_filtered.append(data[i][0])

        return data_filtered

    def load_speed_2(self):
        txt_file = os.path.join(self.gt_data, 'video_name_mapping.txt')
        data = np.genfromtxt(txt_file, delimiter='\t', dtype='str')
        data_filtered = []
        for i in range(len(data)):
            if data[i][1].split('_')[-1] == '2':
                data_filtered.append(data[i][0])

        return data_filtered

    def load_speed_3(self):
        txt_file = os.path.join(self.gt_data, 'video_name_mapping.txt')
        data = np.genfromtxt(txt_file, delimiter='\t', dtype='str')
        data_filtered = []
        for i in range(len(data)):
            if data[i][1].split('_')[-1] == '3':
                data_filtered.append(data[i][0])

        return data_filtered

    def load_human_class(self):
        txt_file = os.path.join(self.gt_data, 'video_name_mapping.txt')
        data = np.genfromtxt(txt_file, delimiter='\t', dtype='str')
        data_filtered = []
        for i in range(len(data)):
            if int(data[i][0]) % 54 < 36:
                if 'angry_girl' in data[i][1] or 'ferbibliotecario' in data[i][1]:
                    data_filtered.append(data[i][0])

        return data_filtered

    def load_car_robot_class(self):
        txt_file = os.path.join(self.gt_data, 'video_name_mapping.txt')
        data = np.genfromtxt(txt_file, delimiter='\t', dtype='str')
        data_filtered = []
        for i in range(len(data)):
            if int(data[i][0]) % 54 < 36:
                if 'robot' in data[i][1] or 'toy_car' in data[i][1]:
                    data_filtered.append(data[i][0])

        return data_filtered

    def load_dog_cat_class(self):
        txt_file = os.path.join(self.gt_data, 'video_name_mapping.txt')
        data = np.genfromtxt(txt_file, delimiter='\t', dtype='str')
        data_filtered = []
        for i in range(len(data)):
            if int(data[i][0]) % 54 < 36:
                if 'shiba' in data[i][1] or 'cat' in data[i][1]:
                    data_filtered.append(data[i][0])

        return data_filtered


def main():
    flag_names = {0: 'calculate All',
                  1: 'calculate Single class',
                  2: 'calculate Multi class',
                  3: 'calculate Speed 1',
                  4: 'calculate Speed 2',
                  5: 'calculate Speed 3',
                  6: 'calculate Human classes',
                  7: 'calculate toy car/robot classes',
                  8: 'calculate dog/cat classe'}
    print(flag_names[args.flag])
    gt_data = args.gt_data
    pred_data = args.pred_data
    flag = args.flag
    metric_data = MetricDataset(pred_data, gt_data, flag)
    metric_dataloader = dataloader.DataLoader(metric_data, batch_size=256)

    iou_meter = AverageValueMeter()
    precision_meter = AverageValueMeter()
    recall_meter = AverageValueMeter()

    for pred, gt in tqdm(metric_dataloader):
        pred.to(device)
        gt.to(device)

        curr_iou = iou(pred, gt)
        iou_meter.add(torch.sum(curr_iou).cpu(
        ).detach().numpy(), curr_iou.shape[0])

        precision, recall, f1 = prf_metrics(pred, gt)

        precision_meter.add(torch.sum(precision).cpu(
        ).detach().numpy(), precision.shape[0])

        recall_meter.add(torch.sum(recall).cpu(
        ).detach().numpy(), recall.shape[0])

        # f1_meter.add(torch.sum(f1).cpu().detach().numpy(), f1.shape[0])
        # print(precision, recall, f1)
        # print(precision)
        # print(precision.shape)

    print("final IOU mean", iou_meter.mean)
    print("final mean precision ", precision_meter.mean)
    print("final mean recall", recall_meter.mean)
    # print("final mean f1", f1_meter.mean)

    # save logging
    safe_mkdir('./detection_results')
    logging.basicConfig(level=logging.DEBUG, filename="./detection_results/logfile", filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    logging.info(f"final IOU mean {iou_meter.mean}")
    logging.info(f"final mean precision {precision_meter.mean}")
    logging.info(f"final mean recall {recall_meter.mean}")
    # TODO: subdirect and exp name for each call or a speration line for each logging entry
    # TODO: demo: example dataset


if __name__ == "__main__":
    main()
