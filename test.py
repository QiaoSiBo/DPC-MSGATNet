import argparse
import numpy as np
import lib
import torch
import os
import logging
from datetime import datetime
from lib.logger import setlogger
from pathlib import Path
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from PIL import Image
from maskToColor import mask_to_vis
from metrics import cal_IOU, to_one_hot, get_DC, get_JS, get_F1



parser = argparse.ArgumentParser(description='DPC-MSGATNet')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run(default: 1)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=1, type=int,
                    metavar='N', help='batch size (default: 8)')
parser.add_argument('--learning_rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate (default: 0.001)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--train_dataset',  type=str)
parser.add_argument('--val_dataset', type=str)
parser.add_argument('--save_freq', type=int, default=5)
parser.add_argument('--modelname', default='off', type=str,
                    help='name of the model to load')
parser.add_argument('--cuda', default="on", type=str, 
                    help='switch on/off cuda option (default: off)')

parser.add_argument('--direc', default='./test_img_pth', type=str,
                    help='directory to save')
parser.add_argument('--logdir', default='./test_img_pth', type=str)
parser.add_argument('--crop', type=int, default=None)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--loaddirec', default='load', type=str)
parser.add_argument('--imgsize', type=int, default=None)
parser.add_argument('--blocksize', type=int, default=None)
parser.add_argument('--gray', default='no', type=str)
parser.add_argument('--mode', default='single', type=str)
args = parser.parse_args()


gray_ = args.gray
direc = args.direc
modelname = args.modelname
imgsize = args.imgsize
loaddirec = args.loaddirec
mode = args.mode
blocksize = args.blocksize
logdir = args.logdir

#############saving the training logs
pth_name = Path(loaddirec).stem  ###the name of the saved pth
###


# if not save_dir.exists():
#     save_dir.mkdir(parents=True)

#set the logger
save_dir = Path(logdir)
setlogger(save_dir.joinpath('test.log'))

if gray_ == "yes":
    from utils_gray import JointTransform2D, ImageToImage2D
    imgchant = 1
else:
    from utils import JointTransform2D, ImageToImage2D
    imgchant = 3

if args.crop is not None:
    crop = (args.crop, args.crop)
else:
    crop = None


tf_val = JointTransform2D(img_size=imgsize, mode='val')
val_dataset = ImageToImage2D(args.val_dataset, tf_val)
valloader = DataLoader(val_dataset, 1, shuffle=True)

device = torch.device("cuda")


if modelname == "msgatnet":
    model = lib.models.msgatnet.MSGAN(img_size=imgsize, imgchan=imgchant)


if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model, device_ids=[0, 1]).cuda()

model.to(device)

if mode == 'multiple':
    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(loaddirec).items()})
if mode == 'single':
    model.load_state_dict(torch.load(loaddirec))

model.eval()

f1 = 0.0
js = 0.0
dc = 0.0
miou = 0.0
nums = 0.0
for batch_idx, (X_batch, y_batch, *rest) in enumerate(valloader):
    # print(batch_idx)
    if isinstance(rest[0][0], str):
                image_filename = rest[0][0]
    else:
                image_filename = '%s.png' % str(batch_idx + 1).zfill(3)

    X_batch = X_batch.to(device='cuda')
    y_batch = y_batch.to(device='cuda')

    y_out = model(X_batch)
    y_out = F.softmax(y_out, dim=1)
    nums += 1
    y_batch_one_hot = to_one_hot(y_out, y_batch)
    miou += cal_IOU(y_out, y_batch_one_hot)
    f1 += get_F1(y_out, y_batch_one_hot)
    dc += get_DC(y_out, y_batch_one_hot)

    y_out = torch.argmax(y_out, dim=1)  # y_out.shape: [1, 224, 224]
    y_out = y_out.squeeze(0)            # y_out.shape: [224, 224]
    ####y_batch
    y_batch = y_batch.squeeze(0)
    y_batch = y_batch.squeeze(0)

    ####Transfer tensor to numpy
    y_pre = y_out.detach().cpu().numpy()
    y_real = y_batch.detach().cpu().numpy()

    del X_batch, y_batch, y_out

    fulldir_val = direc+"/{}/".format("val")
    fulldir_pre = direc+"/{}/".format("pre")
    # print(fulldir+image_filename)
    if not os.path.isdir(fulldir_val):

        os.makedirs(fulldir_val)

    if not os.path.isdir(fulldir_pre):

        os.makedirs(fulldir_pre)
    ####saving the Color Mask
    #y_pre_array = mask_to_vis(5, y_pre)
    #y_real_array = mask_to_vis(5, y_real)
    ##########################save the mask
    y_pre_img = Image.fromarray(y_pre.astype(np.uint8)).convert('P')
    y_real_img = Image.fromarray(y_real.astype(np.uint8)).convert('P')
    #y_pre_img = Image.fromarray(y_pre_array)
    #y_real_img = Image.fromarray(y_real_array)
    y_pre_img.save(fulldir_pre+image_filename+'.png')
    y_real_img.save(fulldir_val+image_filename+'.png')


f1 = f1 / nums
dc = dc / nums
miou = miou / nums
logging.info("----------------------Test logs-------------------------")
logging.info("The saved pth: {}, F1 score: {}".format(pth_name, f1))
logging.info("The saved pth: {}, MIoU score: {}".format(pth_name, miou))
logging.info("The saved pth: {}, DC score: {}".format(pth_name, dc))
logging.info("--------------------------------------------------------")


del f1, miou, dc
