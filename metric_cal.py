import os
from tqdm import tqdm
from PIL import Image
import argparse
import torch
import numpy as np
from utils.metrics import *
from torch.utils.data import Dataset
import torch.utils.data.dataloader as dataloader
from torch.utils.data.dataset import Dataset
from utils.meter import AverageValueMeter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: link parser
parser = argparse.ArgumentParser(description='PyTorch Metrics Calculator')
parser.add_argument('--gt_data', metavar='DIR', help='path to ground truth dataset')
parser.add_argument('--pred_data', metavar='DIR', help='path to predicted dataset')
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
        print("flag:", flag)

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
            gt_folder = os.path.join(self.gt_data,'habitat_sim_DAVIS/Annotations/480p', scene_name)
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
        data = np.genfromtxt(txt_file, delimiter='\t',dtype='str')
        data_filtered = []
        for i in range(len(data)):
            data_filtered.append(data[i][0])

        return data_filtered

    def load_single_class(self):
        # load all data path that has single class stored in .txt file in gt_data
        # meaning first 36 entries of 54
        txt_file = os.path.join(self.gt_data, 'video_name_mapping.txt')
        data = np.genfromtxt(txt_file, delimiter='\t',dtype='str')
        data_filtered = []
        for i in range(len(data)):
            if int(data[i][0]) % 54 < 36:
                data_filtered.append(data[i][0])

        return data_filtered

    def load_multi_class(self):
        # load all data path that has multi class stored in .txt file in gt_data
        # meaning last 18 entries of 54
        txt_file = os.path.join(self.gt_data, 'video_name_mapping.txt')
        data = np.genfromtxt(txt_file, delimiter='\t',dtype='str')
        data_filtered = []
        for i in range(len(data)):
            if int(data[i][0]) % 54 >= 36:
                data_filtered.append(data[i][0])

        return data_filtered

    def load_speed_1(self):
        txt_file = os.path.join(self.gt_data, 'video_name_mapping.txt')
        data = np.genfromtxt(txt_file, delimiter='\t',dtype='str')
        data_filtered = []
        for i in range(len(data)):
            if data[i][1].split('_')[-1] == '1':
                data_filtered.append(data[i][0])

        return data_filtered

    def load_speed_2(self):
        txt_file = os.path.join(self.gt_data, 'video_name_mapping.txt')
        data = np.genfromtxt(txt_file, delimiter='\t',dtype='str')
        data_filtered = []
        for i in range(len(data)):
            if data[i][1].split('_')[-1] == '2':
                data_filtered.append(data[i][0])

        return data_filtered

    def load_speed_3(self):
        txt_file = os.path.join(self.gt_data, 'video_name_mapping.txt')
        data = np.genfromtxt(txt_file, delimiter='\t',dtype='str')
        data_filtered = []
        for i in range(len(data)):
            if data[i][1].split('_')[-1] == '3':
                data_filtered.append(data[i][0])

        return data_filtered

    def load_human_class(self):
        txt_file = os.path.join(self.gt_data, 'video_name_mapping.txt')
        data = np.genfromtxt(txt_file, delimiter='\t',dtype='str')
        data_filtered = []
        for i in range(len(data)):
            if int(data[i][0]) % 54 < 36:
                if 'angry_girl' in data[i][1] or 'ferbibliotecario' in data[i][1]:
                    data_filtered.append(data[i][0])

        return data_filtered

    def load_car_robot_class(self):
        txt_file = os.path.join(self.gt_data, 'video_name_mapping.txt')
        data = np.genfromtxt(txt_file, delimiter='\t',dtype='str')
        data_filtered = []
        for i in range(len(data)):
            if int(data[i][0]) % 54 < 36:
                if 'robot' in data[i][1] or 'toy_car' in data[i][1]:
                    data_filtered.append(data[i][0])

        return data_filtered

    def load_dog_cat_class(self):
        txt_file = os.path.join(self.gt_data, 'video_name_mapping.txt')
        data = np.genfromtxt(txt_file, delimiter='\t',dtype='str')
        data_filtered = []
        for i in range(len(data)):
            if int(data[i][0]) % 54 < 36:
                if 'shiba' in data[i][1] or 'cat' in data[i][1]:
                    data_filtered.append(data[i][0])

        return data_filtered

def main():
    gt_data = "/home/gao/dev/project_remote/Habitat-sim-ext/randomwalk/output/habitat_sim_excl_static_30scenes_newPitch_originalModel"
    # gt_data = "/home/gao/dev/project_remote/Habitat-sim-ext/randomwalk/output/habitat_sim_excl_static_30scenes_newPitch_originalModel"
    pred_data = "/home/gao/dev/project_remote/Habitat-sim-ext/randomwalk/output/cis_anno/habitatDyn_dynamic_30scenes_new"
    metric_data = MetricDataset(pred_data, gt_data, 8)
    metric_dataloader = dataloader.DataLoader(metric_data, batch_size=256)

    # TODO modify to true per category
    iou_meter = AverageValueMeter()
    precision_meter = AverageValueMeter()
    recall_meter = AverageValueMeter()

    for pred, gt in tqdm(metric_dataloader):
        pred.to(device)
        gt.to(device)

        curr_iou = iou(pred, gt)
        iou_meter.add(torch.sum(curr_iou).cpu().detach().numpy(), curr_iou.shape[0])

        precision, recall, f1 = prf_metrics(pred, gt)

        precision_meter.add(torch.sum(precision).cpu().detach().numpy(), precision.shape[0])

        recall_meter.add(torch.sum(recall).cpu().detach().numpy(), recall.shape[0])

        # f1_meter.add(torch.sum(f1).cpu().detach().numpy(), f1.shape[0])
        # print(precision, recall, f1)
        # print(precision)
        # print(precision.shape)

    print("final IOU mean", iou_meter.mean)
    print("final mean precision ", precision_meter.mean)
    print("final mean recall", recall_meter.mean)
    # print("final mean f1", f1_meter.mean)

    # TODO: save .npy
    # TODO: demo: example dataset + result in a md file, 介绍过程

if __name__ == "__main__":
    main()