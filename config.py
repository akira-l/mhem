import os
import pandas as pd
import torch

from transforms import transforms
from utils.autoaugment import ImageNetPolicy

if os.path.exists('../pretrained'):
    pretrained_model = {'resnet50' : '../pretrained/resnet50-19c8e357.pth',
                        'resnet101': './models/pretrained/se_resnet101-7e38fcc6.pth',
                        'senet154':'./models/pretrained/checkpoint_epoch_017_prec3_93.918_pth.tar'}
else:
    pretrained_model = {'resnet50' : './pretrained/resnet50-19c8e357.pth',
                        'resnet101': './models/pretrained/se_resnet101-7e38fcc6.pth',
                        'senet154':'./models/pretrained/checkpoint_epoch_017_prec3_93.918_pth.tar'}


def load_data_transformers(resize_reso=512, crop_reso=448, swap_num=[7, 7]):
    center_resize = 600
    Normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    data_transforms = {
       	'swap': transforms.Compose([
            transforms.Resize((resize_reso, resize_reso)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomCrop((crop_reso, crop_reso)),
            transforms.RandomHorizontalFlip(),
            transforms.Randomswap((swap_num[0], swap_num[1])),
        ]),
       	'food_swap': transforms.Compose([
            transforms.Resize((resize_reso, resize_reso)),
            transforms.RandomRotation(degrees=90),
            #transforms.RandomCrop((crop_reso, crop_reso)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size=crop_reso, scale=(0.75, 1)),
            transforms.Randomswap((swap_num[0], swap_num[1])),
        ]),
       	'food_unswap': transforms.Compose([
            transforms.Resize((resize_reso, resize_reso)),
            transforms.RandomRotation(degrees=90),
            #transforms.RandomCrop((crop_reso, crop_reso)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomResizedCrop(size=crop_reso, scale=(0.75, 1)),
        ]),

        'unswap': transforms.Compose([
            transforms.Resize((resize_reso, resize_reso)),
            transforms.RandomRotation(degrees=15),
            #transforms.RandomCrop((crop_reso,crop_reso), padding=8),
            transforms.RandomResizedCrop(size=crop_reso, scale=(0.6, 1)),
            #transforms.RandomCrop((crop_reso,crop_reso)),
            transforms.RandomHorizontalFlip(),
        ]),

        'train_totensor': transforms.Compose([
            transforms.Resize((crop_reso, crop_reso)),
            #ImageNetPolicy(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'val_totensor': transforms.Compose([
            #transforms.Resize((crop_reso, crop_reso)),
            transforms.Resize((resize_reso, resize_reso)),
            transforms.CenterCrop((crop_reso, crop_reso)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'test_totensor': transforms.Compose([
            transforms.Resize((resize_reso, resize_reso)),
            transforms.CenterCrop((crop_reso, crop_reso)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'None': None,
        'Centered_swap': transforms.Compose([
            transforms.CenterCrop((center_resize, center_resize)),
            transforms.Resize((resize_reso, resize_reso)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomCrop((crop_reso, crop_reso)),
            transforms.RandomHorizontalFlip(),
            transforms.Randomswap((swap_num[0], swap_num[1])),
       ]),
        'Centered_unswap': transforms.Compose([
            transforms.CenterCrop((center_resize, center_resize)),
            transforms.Resize((resize_reso, resize_reso)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomCrop((crop_reso, crop_reso)),
            transforms.RandomHorizontalFlip(),
       ]),
        'Tencrop': transforms.Compose([
            transforms.Resize((resize_reso, resize_reso)),
            transforms.TenCrop((crop_reso, crop_reso)),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
      ])
    }

    return data_transforms




class LoadConfig(object):
    def __init__(self, args, version):
        if version == 'train':
            get_list = ['train', 'val']
        elif version == 'val':
            get_list = ['val']
        elif version == 'test':
            get_list = ['test']
        elif version == 'ensamble':
            get_list = []
        else:
            raise Exception("train/val/test ???\n")

        ###############################
        #### add dataset info here ####
        ###############################

        if args.dataset == 'CUB':
            self.dataset = args.dataset
            self.rawdata_root = './../../data/CUB_200_2011/all'
            self.anno_root = './../../data/CUB_200_2011'
            self.numcls = 200

        if args.dataset == 'stcar':
            self.dataset = args.dataset
            self.rawdata_root = './../dataset/stcar'
            self.anno_root = './../dataset/stcar'
            self.numcls = 196


        if args.dataset == 'aircraft':
            self.dataset = args.dataset
            self.rawdata_root = './../dataset/aircraft'
            self.anno_root = './../dataset/aircraft/anno'
            self.numcls = 100


        if 'train' in get_list:
            self.train_anno = pd.read_csv(os.path.join(self.anno_root, 'train.txt'),\
                                           sep=" ",\
                                           header=None,\
                                           names=['ImageName', 'label'])

        if 'val' in get_list:
            self.val_anno = pd.read_csv(os.path.join(self.anno_root, 'val.txt'),\
                                           sep=" ",\
                                           header=None,\
                                           names=['ImageName', 'label'])

        if 'test' in get_list:
            self.test_anno = pd.read_csv(os.path.join(self.anno_root, 'test_labeled.txt'),\
                                           sep=" ",\
                                           header=None,\
                                           names=['ImageName', 'label'])

        self.swap_num = args.swap_num

        self.save_dir = './net_model'
        self.backbone = args.backbone
        self.train_bs = args.train_batch

        self.use_backbone = True

        self.weighted_sample = False

        self.log_folder = './logs'
        if not os.path.exists(self.log_folder):
            os.mkdir(self.log_folder)


