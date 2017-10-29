import os
import os.path
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

#if sys.version_info[0] == 2:
#    import xml.etree.cElementTree as ET
#else:
#    import xml.etree.ElementTree as ET


LABELS = ['species_fourspot',
           'species_grey sole',
           'species_other', 
           'species_plaice',
           'species_summer',
           'species_windowpane',
           'species_winter']

LABELS_2_IND = {
   'species_fourspot' : 0,
   'species_grey sole' : 1,
   'species_other' : 2,
   'species_plaice' : 3,
   'species_summer': 4,
   'species_windowpane' : 5,
   'species_winter' : 6 
}

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

class FISHdetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, image_paths, image_annots, transform=None, dataset_name='fish_detection'):
        
        # root
        # image_sets
        
        # self.root = root
        self.image_paths = image_paths
        self.image_annots = image_annots
        self.transform = transform
        # self.target_transform = target_transform
        
        self.name = dataset_name
        # self._annopath = os.path.join('%s', 'Annotations', '%s.xml')
        # self._imgpath = os.path.join('%s', 'JPEGImages', '%s.jpg')

        # self.ids = list()
        # for (year, name) in image_sets:
        #    rootpath = os.path.join(self.root, 'VOC' + year)
        #    for line in open(os.path.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
        #        self.ids.append((rootpath, line.strip()))

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.image_paths)

    def pull_item(self, index):
        # img_id = self.ids[index]
        
        img_path = self.image_paths[index]
        # target = ET.parse(self._annopath % img_id).getroot()
        target = self.image_annots[index]
        target = np.asarray(target).reshape(-1,5)
        
        try:
            img = cv2.imread(img_path)
            height, width, channels = img.shape
        except:
            img = cv2.imread('random_image.jpg')
            height, width, channels = img.shape            
            
        #if self.target_transform is not None:
        #    target = self.target_transform(target, width, height)


        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        # img_id = self.ids[index]
        img_path = self.image_paths[index]
        
        return cv2.imread(img_path, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        # img_id = self.ids[index]
        #anno = ET.parse(self._annopath % img_id).getroot()
        # gt = self.target_transform(anno, 1, 1)
        img_path = self.image_paths[index]
        # target = ET.parse(self._annopath % img_id).getroot()
        target = self.image_annots[index]
        
        return img_path, target

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets
