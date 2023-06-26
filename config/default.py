#!/usr/bin/env python3

class SEMANTIC_ANTICIPATOR:
    def __init__(self, raw):
        self.type = raw['type']
        self.resnet_type = raw['resnet_type']
        self.unet_nsf = raw['unet_nsf']
        self.map_scale = raw['map_scale']
        self.nclasses = raw['nclasses']
        self.freeze_features = raw['freeze_features']
        self.map_size = raw['map_size']
        self.imgh = raw['imgh']
        self.imgw = raw['imgw']
        self.maxmium_object_num = raw['maxmium_object_num']
        self.device = raw['device']

class OUTPUT_NORMALIZATION:
    def __init__(self, raw):
        self.channel_0 = raw['channel_0']
        self.channel_1 = raw['channel_1']

class IMG_NORMALIZATION:
    def __init__(self, raw):
        self.res18_img_mean = raw['res18_img_mean']
        self.res18_img_std = raw['res18_img_std']
        self.focal_img_mean = raw['focal_img_mean']
        self.focal_img_std = raw['focal_img_std']        

class Config:
    def __init__(self, raw):
        self.SEMANTIC_ANTICIPATOR = SEMANTIC_ANTICIPATOR(raw['SEMANTIC_ANTICIPATOR'])
        self.OUTPUT_NORMALIZATION = OUTPUT_NORMALIZATION(raw['OUTPUT_NORMALIZATION'])
        self.IMG_NORMALIZATION = IMG_NORMALIZATION(raw['IMG_NORMALIZATION'])