# Code for MSGAN
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import lib
import argparse
import logging
from datetime import datetime
from torch import nn
from torch.utils.data import DataLoader
from PIL import Image
from lib.logger import setlogger
import torch.nn.functional as F
import numpy as np
import torch
from metrics import jaccard_index, f1_score, LogNLLLoss, classwise_f1, classwise_iou, cal_IOU, to_one_hot, get_DC, get_JS
from lib.loss_functions import SoftDiceLoss, DC_and_CE_loss
from lib.nd_softmax import softmax_helper
from torch.optim import lr_scheduler
from maskToColor import colormap, mask_to_vis
import cv2

parser = argparse.ArgumentParser(description='DPC-MSGATNet')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=400, type=int, metavar='N',
                    help='number of total epochs to run(default: 400)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=1, type=int,
                    metavar='N', help='batch size (default: 1)')
parser.add_argument('--learning_rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate (default: 0.001)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-5)')
parser.add_argument('--train_dataset', required=True, type=str)
parser.add_argument('--val_dataset', type=str)
parser.add_argument('--save_freq', type=int,default = 10)

parser.add_argument('--modelname', default='msgan', type=str,
                    help='type of model')
parser.add_argument('--cuda', default="on", type=str, 
                    help='switch on/off cuda option (default: off)')
parser.add_argument('--aug', default='off', type=str,
                    help='turn on img augmentation (default: False)')
parser.add_argument('--load', default='default', type=str,
                    help='load a pretrained model')
parser.add_argument('--save', default='default', type=str,
                    help='save the model')
parser.add_argument('--direc', default='./medt', type=str,
                    help='directory to save')
parser.add_argument('--crop', type=int, default=None)
parser.add_argument('--imgsize', type=int, default=None)
parser.add_argument('--blocksize', type=int, default=None)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--gray', default='no', type=str)



args = parser.parse_args()
gray_ = args.gray
aug = args.aug
direc = args.direc
modelname = args.modelname
imgsize = args.imgsize
blocksize = args.blocksize



#############saving the training logs
sub_dir = modelname + '_' + datetime.strftime(datetime.now(), '%m%d-%H%M%S')
save_dir = os.path.join(direc, sub_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

#set the logger
setlogger(os.path.join(save_dir, 'train.log'))

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
###Get the Transform to the dataset
tf_train = JointTransform2D(img_size=imgsize, mode='train')
tf_val = JointTransform2D(img_size=imgsize, mode='val')
###Get the dataset
train_dataset = ImageToImage2D(args.train_dataset, tf_train)
val_dataset = ImageToImage2D(args.val_dataset, tf_val)
##Get the dataloader
dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
valloader = DataLoader(val_dataset, 1, shuffle=True)

device = torch.device("cuda")


if modelname == "msgatnet":
    model = lib.models.msgatnet.MSGAN(img_size=imgsize, block_size=blocksize, imgchan=imgchant)


if torch.cuda.device_count() > 1:
  logging.info("Let's use {} GPUs!".format(torch.cuda.device_count()))
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model, device_ids=[0, 1]).cuda()

model.to(device)




#criterion = LogNLLLoss()
#apply_nonlin = softmax_helper
criterion = DC_and_CE_loss(soft_dice_kwargs={'batch_dice': True, 'smooth': 1e-5, 'do_bg': False})
#criterion = SoftDiceLoss(apply_nonlin=apply_nonlin, batch_dice=True, smooth=1e-5, do_bg=False)
optimizer = torch.optim.Adam(list(model.parameters()), lr=args.learning_rate,
                              weight_decay=1e-5)
#optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8,
                                               patience=15,
                                               verbose=True, threshold=1e-4,
                                               threshold_mode="rel")

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logging.info("Total_params: {}".format(pytorch_total_params))

###

seed = 3000
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# torch.set_deterministic(True)
# random.seed(seed)


for epoch in range(args.epochs):

    epoch_running_loss = 0
    logging.info("Epoch: {}".format(epoch))
    model.train()
    for batch_idx, (X_batch, y_batch, *rest) in enumerate(dataloader):        
        
        

        X_batch = X_batch.to(device ='cuda')
        y_batch = y_batch.to(device='cuda')


        
        # ===================forward=====================
        

        output = model(X_batch)

        loss = criterion(output, y_batch)
        
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_running_loss += loss.item()

    lr_scheduler.step(epoch_running_loss)
    # ===================log========================
    current_lr = optimizer.state_dict()['param_groups'][0]['lr']
    logging.info('epoch [{}/{}], loss:{:.4f}, current_lr: {:.4f}'.format(epoch, args.epochs, epoch_running_loss/(batch_idx+1), current_lr))

    
    if epoch == 50:
        for param in model.parameters():
            param.requires_grad =True
    if (epoch % args.save_freq) ==0:
        model.eval()
        with torch.no_grad():
            for batch_idx, (X_batch, y_batch, *rest) in enumerate(valloader):
                # print(batch_idx)
                if isinstance(rest[0][0], str):
                            image_filename = rest[0][0]
                            image_filename = '%s.png' % str(image_filename.split('.')[0])
                else:
                            image_filename = '%s.png' % str(batch_idx + 1).zfill(3)

                X_batch = X_batch.to(device='cuda')
                y_batch = y_batch.to(device='cuda')
                # start = timeit.default_timer()
                y_out = model(X_batch)
                # stop = timeit.default_timer()
                # print('Time: ', stop - start)
                y_out = F.softmax(y_out, dim=1)
                y_out = torch.argmax(y_out, dim=1)
                y_out = y_out.squeeze(0)
                ####y_batch
                y_batch = y_batch.squeeze(0)
                y_batch = y_batch.squeeze(0)

                ####Transfer tensor to numpy
                y_pre = y_out.detach().cpu().numpy()
                y_real = y_batch.detach().cpu().numpy()

                del X_batch, y_batch, y_out

                fulldir_val = direc+"/{}/{}/".format(epoch, "val")
                fulldir_pre = direc+"/{}/{}/".format(epoch, "pre")
                # print(fulldir+image_filename)
                if not os.path.isdir(fulldir_val):

                    os.makedirs(fulldir_val)

                if not os.path.isdir(fulldir_pre):

                    os.makedirs(fulldir_pre)
                ####saving the Color Mask
                y_pre_array = mask_to_vis(5, y_pre)
                y_real_array = mask_to_vis(5, y_real)
                y_pre_img = Image.fromarray(y_pre_array)
                y_real_img = Image.fromarray(y_real_array)
                y_pre_img.save(fulldir_pre+image_filename)
                y_real_img.save(fulldir_val+image_filename)

            fulldir = direc+"/{}/".format(epoch)
            torch.save(model.state_dict(), fulldir+args.modelname+".pth")
            #torch.save(save_pth, direc+"/"+"final_model.pth")
            


