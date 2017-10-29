import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import argparse
from torch.autograd import Variable
import torch.utils.data as data
from data import v2, v1, AnnotationTransform, VOCDetection, detection_collate, VOC_CLASSES
from data import FISHdetection
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import numpy as np
import time

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

VOCroot = '../data/VOCdevkit/'

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
parser.add_argument('--version', default='v2', help='conv11_2(v2) or pool6(v1) as last layer')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth', help='pretrained base model')
parser.add_argument('--jaccard_threshold', default=0.5, type=float, help='Min Jaccard index for matching')
parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training')
parser.add_argument('--resume', default=None, type=str, help='Resume from checkpoint')
parser.add_argument('--num_workers', default=2, type=int, help='Number of workers used in dataloading')
parser.add_argument('--iterations', default=120000, type=int, help='Number of training iterations')
parser.add_argument('--start_iter', default=0, type=int, help='Begin counting iterations starting from this value (should be used with resume)')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--log_iters', default=True, type=bool, help='Print the loss at each iteration')
parser.add_argument('--visdom', default=False, type=str2bool, help='Use visdom to for loss visualization')
parser.add_argument('--send_images_to_visdom', type=str2bool, default=False, help='Sample a random image from each 10th batch, send it to visdom after augmentations step')
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--voc_root', default=VOCroot, help='Location of VOC root directory')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--logname', default='ssd_fish_test', type=str, help='Name for saving your logs')

args = parser.parse_args()

print ('Check the args are passed correctly!')
print(args)

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

cfg = (v1, v2)[args.version == 'v2']

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

import pandas as pd
import os
import glob 
import numpy as np
import random 
from sklearn.model_selection import train_test_split

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

ANNOT_LIST = ['x1_new','y1_new','x2_new','y2_new','species']
REMOVE_NEGATIVE_ITEMS = 0.995
VALID_SHARE = 0.2

def change_coords(x1,y1,x2,y2,f_len):
    
    max_x = 1280
    max_y = 720
    
    x_av = (x2+x1)/2
    y_av = (y2+y1)/2
    

    if(x_av-f_len/2)>max_x:
        x1_new = max_x
    elif (x_av-f_len/2)<0:
        x1_new = 0
    else:
        x1_new=x_av-f_len/2 
        
    if(x_av+f_len/2)>max_x:
        x2_new = max_x
    else:
        x2_new=x_av+f_len/2 
        
    if(y_av-f_len/2)>max_y:
        y1_new = max_y
    elif (y_av-f_len/2)<0:
        y1_new = 0
    else:
        y1_new=y_av-f_len/2 

    if(y_av+f_len/2)>max_y:
        y2_new = max_y
    else:
        y2_new=y_av+f_len/2

    return x1_new,y1_new,x2_new,y2_new

df = pd.read_csv('../fish-video/train.csv')
df['_'] = df.apply(lambda row: change_coords(row['x1'],row['y1'],row['x2'],row['y2'],row['length']), axis=1)
df[['x1_new','y1_new','x2_new','y2_new']] = df['_'].apply(pd.Series)
del df['_']

cpt = {r:len(files) for r, d, files in os.walk('../extra_space/train_pics/')}
del cpt['../extra_space/train_pics/']
cpt = {k.split('/')[3]: v for k, v in cpt.items()}
cpt_zero = {k: v for k, v in cpt.items() if v == 0}
cpt_non_zero = {k: v for k, v in cpt.items() if v > 0}

species_list = LABELS

df['species'] = df[species_list].idxmax(axis=1).apply(lambda x: int(LABELS_2_IND[x]))
df.loc[df.fish_number.isnull()==True, 'species'] = np.nan
df['path'] = df.apply(lambda row: '../extra_space/train_pics/{}/{}.jpg'.format(row['video_id'],str(row['frame']+1).zfill(4)), axis=1)

imgs_fish = set(list(df[df['fish_number'].notnull()].path.values))
 # [[xmin, ymin, xmax, ymax, label_ind], ... ]

img_annots = list ( df[ (df['fish_number'].notnull()) & (~df.video_id.isin(list(cpt_zero.keys())))][ANNOT_LIST].values.tolist() )
img_paths = list ( df[ (df['fish_number'].notnull()) & (~df.video_id.isin(list(cpt_zero.keys())))]['path'].values.tolist() )    

VALID_SHARE = 0.2

train_img_paths, valid_img_paths, train_img_annots, valid_img_annots = train_test_split(img_paths,img_annots, test_size=VALID_SHARE, random_state=42)

del img_annots,img_paths
del df    
    

# train_sets = 'train'
ssd_dim = 300  # only support 300 now
means = (104, 117, 123)  # only support voc now

# num_classes = len(VOC_CLASSES) + 1

# w background class?
num_classes = len(LABELS)+1
print('Number of classes is {}'.format(num_classes) )

batch_size = args.batch_size
accum_batch_size = 32
iter_size = accum_batch_size / batch_size
max_iter = 100000
weight_decay = 0.0005
stepvalues = (5000, 15000, 50000)
gamma = 0.1
momentum = 0.9

if args.visdom:
    import visdom
    viz = visdom.Visdom()

ssd_net = build_ssd('train', 300, num_classes)
net = ssd_net

if args.cuda:
    net = torch.nn.DataParallel(ssd_net)
    cudnn.benchmark = True

if args.resume:
    print('Resuming training, loading {}...'.format(args.resume))
    ssd_net.load_weights(args.resume)
else:
    vgg_weights = torch.load(args.save_folder + args.basenet)
    print('Loading base network...')
    ssd_net.vgg.load_state_dict(vgg_weights)

if args.cuda:
    net = net.cuda()


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


if not args.resume:
    print('Initializing weights...')
    # initialize newly added layers' weights with xavier method
    ssd_net.extras.apply(weights_init)
    ssd_net.loc.apply(weights_init)
    ssd_net.conf.apply(weights_init)

# optimizer = optim.SGD(net.parameters(), lr=args.lr,
#                      momentum=args.momentum, weight_decay=args.weight_decay)

optimizer = optim.Adam(net.parameters(), lr=args.lr,weight_decay=args.weight_decay)

criterion = MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False, args.cuda)


def train():
    net.train()
    # loss counters
    loc_loss = 0  # epoch
    conf_loss = 0
    epoch = 0
    print('Loading Dataset...')

    # dataset = VOCDetection(args.voc_root, train_sets, SSDAugmentation(
    #    ssd_dim, means), AnnotationTransform())

    dataset = FISHdetection(image_paths=train_img_paths,
                            image_annots=train_img_annots,
                            transform=SSDAugmentation(ssd_dim, means),
                            dataset_name='fish_detection')

    val_dataset = FISHdetection(image_paths=valid_img_paths,
                            image_annots=valid_img_annots,
                            transform=SSDAugmentation(ssd_dim, means),
                            dataset_name='fish_detection_valid')
    
    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on', dataset.name)
    print ('The length of the dataset is {}, epoch size is {} batches'.format(len(dataset), epoch_size))
    
    
    step_index = 0
    if args.visdom:
        # initialize visdom loss plot
        lot = viz.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1, 3)).cpu(),
            opts=dict(
                xlabel='Iteration',
                ylabel='Loss',
                title='Current SSD Training Loss',
                legend=['Loc Loss', 'Conf Loss', 'Loss']
            )
        )
        epoch_lot = viz.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1, 2)).cpu(),
            opts=dict(
                xlabel='Epoch',
                ylabel='Loss',
                title='Epoch SSD Train / Val loss',
                legend=['Train Loss', 'Valid Loss']
            )
        )
    batch_iterator = None
    #data_loader = data.DataLoader(dataset, batch_size, num_workers=args.num_workers,
    #                              shuffle=True, collate_fn=detection_collate, pin_memory=True)
    
    data_loader = data.DataLoader(dataset, batch_size, num_workers=args.num_workers,
                                      shuffle=True, collate_fn=detection_collate, pin_memory=True)    
    
    val_data_loader = data.DataLoader(val_dataset, batch_size, num_workers=args.num_workers,
                                      shuffle=True, collate_fn=detection_collate, pin_memory=True)    
        
    loss_avm = AverageMeter()
    
    for iteration in range(args.start_iter, max_iter):
        if (not batch_iterator) or (iteration % epoch_size == 0):
            # create batch iterator
            batch_iterator = iter(data_loader)
        if iteration in stepvalues:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)
            """
            if args.visdom:
                viz.line(
                    X=torch.ones((1, 3)).cpu() * epoch,
                    Y=torch.Tensor([loc_loss, conf_loss,
                        loc_loss + conf_loss]).unsqueeze(0).cpu() / epoch_size,
                    win=epoch_lot,
                    update='append'
                )
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0
            epoch += 1
            """
        # load train data
        images, targets = next(batch_iterator)

        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
        else:
            images = Variable(images)
            targets = [Variable(anno, volatile=True) for anno in targets]
            
        # forward
        t0 = time.time()
        out = net(images)
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        
        # memory leak here?
        loss_avm.update(loss_l.data[0] + loss_c.data[0], images.size(0))

        loss.backward()
        optimizer.step()
        t1 = time.time()
        # loc_loss += loss_l.data[0]
        # conf_loss += loss_c.data[0]

        if iteration % 10 == 0:
            print('Timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data[0]), end=' ')
            if args.visdom and args.send_images_to_visdom:
                random_batch_index = np.random.randint(images.size(0))
                viz.image(images.data[random_batch_index].cpu().numpy())
                
                
        # Do the validation bit
        if (iteration % epoch_size  == 0) and (iteration>100):
            print ('Validation triggered')
            val_loss = validate(val_data_loader, net, criterion)
            
            print ('Train loss is {} vs. Validation loss'.format(loss_avm.avg, val_loss))
            
            if args.visdom:
                viz.line(
                    X=torch.ones((1, 2)).cpu() * epoch,
                    Y=torch.Tensor([loss_avm.avg, val_loss]).unsqueeze(0).cpu(),
                    win=epoch_lot,
                    update='append'
                )
            # reset epoch loss counters
            # loc_loss = 0
            # conf_loss = 0
            epoch += 1            
            loss_avm.reset()

            
        if args.visdom:
            viz.line(
                X=torch.ones((1, 3)).cpu() * iteration,
                Y=torch.Tensor([loss_l.data[0], loss_c.data[0],
                    loss_l.data[0] + loss_c.data[0]]).unsqueeze(0).cpu(),
                win=lot,
                update='append'
            )
            # hacky fencepost solution for 0th epoch plot
            if iteration == 0:
                viz.line(
                    X=torch.zeros((1, 3)).cpu(),
                    Y=torch.Tensor([loc_loss, conf_loss,
                        loc_loss + conf_loss]).unsqueeze(0).cpu(),
                    win=epoch_lot,
                    update=True
                )
        if iteration % 5000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(ssd_net.state_dict(), 'weights/{}'.format(args.logname) +
                       repr(iteration) + '.pth')
    torch.save(ssd_net.state_dict(), args.save_folder + args.logname + args.version + '.pth')

    
def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    # top1 = AverageMeter()
    # top5 = AverageMeter()

    end = time.time()
    # switch to evaluate mode
    model.eval()

    for i, (input, target) in enumerate(val_loader):
        
        # target = torch.stack(target)
        # target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        # images = Variable(images.cuda())
        target_var = [Variable(anno.cuda(), volatile=True) for anno in target]
        
        # target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss_l, loss_c = criterion(output, target_var)

        # measure accuracy and record loss
        #prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        loss_meter.update(loss_l.data[0]+loss_l.data[0], input.size(0))
    
        #top1.update(prec1[0], input.size(0))
        #top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   i, len(val_loader), batch_time=batch_time, loss=loss_meter))
            
    return loss_meter.avg
    

def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    print('lr adjusted by a factor of {}'.format(gamma))
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    train()
