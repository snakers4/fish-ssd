import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data import VOCroot
from data import VOC_CLASSES as labelmap
import torch.utils.data as data

from data import AnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES
from ssd import build_ssd
import pandas as pd

import sys
import os
import time
import argparse
import numpy as np
import pickle
import cv2
from data import FISHdetection
import tqdm 

from data import detection_collate
from utils.augmentations import SSDAugmentation_predict

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

ssd_dim = 300  # only support 300 now
means = (104, 117, 123)  # only support voc now

broken_vids = ['P3QkoeOjxoM6pDKb','tJinkrdMMZ477RGi']

df_sub = pd.read_csv('../fish-video/submission.csv')
df_sub.head()

df_sub['xmin'] = 0
df_sub['xmax'] = 0
df_sub['ymin'] = 0
df_sub['ymax'] = 0

ids_test = [{'row_id':row[0],'frame':row[1],'video_id':row[2]} 
            for row in df_sub[(~df_sub.video_id.isin(broken_vids))][['row_id','frame','video_id']].values]

image_paths = [('../extra_space/test_pics/{}/{}.jpg'.format(frame['video_id'],str(frame['frame']+1).zfill(4))) for frame in ids_test]
image_annots = [([0.1,0.1,0.2,0.2,1.0]) for path in image_paths]

"""
parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/ssd300_mAP_77.43_v2.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--voc_root', default=VOCroot, help='Location of VOC root directory')

args = parser.parse_args()
"""
args = type('test', (), {})()
args.cuda = True

LABELS = ['species_fourspot',
           'species_grey sole',
           'species_other', 
           'species_plaice',
           'species_summer',
           'species_windowpane',
           'species_winter']

dataset_mean = (104, 117, 123)

# load net
num_classes = len(LABELS) + 1 # +1 background

net = build_ssd('train', 300, num_classes)

args.trained_model = 'weights/ssd_fish75000.pth'
args.num_workers = 4

net = build_ssd('test', 300, num_classes) # initialize SSD
net.load_state_dict(torch.load(args.trained_model))
net.eval()

if args.cuda:
    net = net.cuda()
    cudnn.benchmark = True
    
net.eval()

print('Finished loading model!')
# load data
dataset = FISHdetection(image_paths=image_paths,
                        image_annots=image_annots,
                        transform=SSDAugmentation_predict(ssd_dim, means),
                        dataset_name='fish_detection')

# all detections are collected into:
#    all_boxes[cls][image] = N x 5 array of detections in
#    (x1, y1, x2, y2, score)
all_boxes = [[[] for _ in range(len(ids_test))]
             for _ in range(len(LABELS)+1)]

with tqdm.tqdm(total=len(ids_test)) as pbar:
    for i in range(len(ids_test)):
    # for i in range(1000):  
    
        try:
            im, gt, h, w = dataset.pull_item(i)
            x = Variable(im.unsqueeze(0))

            if args.cuda:
                x = x.cuda()

            detections = net(x).data

            # skip j = 0, because it's the background class    
            for j in range(1, detections.size(1)):
                dets = detections[0, j, :]
                mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()     
                dets = torch.masked_select(dets, mask).view(-1, 5)
                if dets.dim() == 0:
                    continue        
                boxes = dets[:, 1:]
                boxes[:, 0] *= w
                boxes[:, 2] *= w
                boxes[:, 1] *= h
                boxes[:, 3] *= h    
                scores = dets[:, 0].cpu().numpy()
                cls_dets = np.hstack((boxes.cpu().numpy(), scores[:, np.newaxis])) \
                    .astype(np.float32, copy=False)
                all_boxes[j][i] = cls_dets
                
        except Exception as e:
            print()
            print('{} was triggered on item {}'.format(str(e), i))
            
        pbar.update(1)
        
np.save('fish_ssd_preds',all_boxes)