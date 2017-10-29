import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

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
from utils.augmentations import SSDAugmentation_predict_fish_crop

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

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

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

args.trained_model = 'weights/fishes_w_validate_r_crop30000.pth'
args.num_workers = 4

ssd_net = build_ssd('test', 300, num_classes)
net = ssd_net

if args.cuda:
    net = torch.nn.DataParallel(ssd_net)
    cudnn.benchmark = True


ssd_net.load_weights(args.trained_model)

if args.cuda:
    net = net.cuda()
    
net.eval()
print('Finished loading model!')

for fish_mode in (1,2,3):
    
    print('Predicting on fish mode {}'.format(fish_mode))
    
    """
    dataset = FISHdetection(image_paths=image_paths,
                            image_annots=image_annots,
                            transform=SSDAugmentation_predict_fish_crop(ssd_dim, means,mode = fish_mode),
                            dataset_name='fish_detection')
    """
    # load data
    dataset = FISHdetection(image_paths=image_paths,
                            image_annots=image_annots,
                            transform=SSDAugmentation_predict_fish_crop(ssd_dim, means,mode = fish_mode),
                            dataset_name='fish_detection')
    batch_size = 128

    data_loader = data.DataLoader(dataset,
                                  batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=False,
                                  collate_fn=detection_collate,
                                  pin_memory=True) 

    
    all_boxes = [[[] for _ in range(len(ids_test))]
                 for _ in range(len(LABELS)+1)]

    with tqdm.tqdm(total=len(ids_test)) as pbar:
        # for i in range(len(ids_test)):
        # for i in range(1000):  
        i = 0
        prev_i = 0
        
        for batch_number, (input, target) in enumerate(data_loader):

            # print(batch_number)
            
            try:
                input_var = torch.autograd.Variable(input, volatile=True)
                output = net(input_var)                    

                # Because we are doing this on crops
                h, w = 720,720

                for output_item in output:

                    # print(h,w)
                    # im, gt, h, w = dataset.pull_item(i)
                    # x = Variable(im.unsqueeze(0))

                    # if args.cuda:
                    # x = x.cuda()

                    # detections = net(x).data
                    detections = output_item.unsqueeze(0).data
                    gt = np.array([[ 0.1,  0.1,  0.2,  0.2,  1. ]])                    

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

                    i += 1
                    
            except Exception as e:
                i = prev_i + input.size(0)
                prev_i += input.size(0)
                print('{} was triggered on mini batch number {}'.format(str(e), batch_number))

            pbar.update(input.size(0))
            
            # if batch_number>0:
            #    break
    # break
    print ('{} items were processes before saving the file'.format(i))
    np.save('fish_ssd_fish_crop_batch_{}'.format(fish_mode),all_boxes)