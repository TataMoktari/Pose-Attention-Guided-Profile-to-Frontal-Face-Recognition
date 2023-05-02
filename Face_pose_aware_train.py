import argparse
# from utils import *
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from Facenet_tune import FacePoseAwareNet
import torch.backends.cudnn as cudnn
from utils import *
from contrastive_dataset_generation import get_dataset
from train_dataset import get_dataset
from validation_dataset import get_dataset_val
import os
from torch import optim
import torch.backends.cudnn as cudnn

#################################################################################################
parser = argparse.ArgumentParser(description='pose aware profile-to-frontal face recognition network')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--lr', default=0.0001, help='learning rate')
parser.add_argument('--margin', default=5, type=int, help='margin')
parser.add_argument('--photo_folder', type=str,
                    default='/home/moktari/Moktari/2022/facenet-pytorch-master/Facenet_Finetune_Pose_aware_Attention/300W_LPA_frontal',
                    help='path to data')
parser.add_argument('--print_folder', type=str,
                    default='/home/moktari/Moktari/2022/facenet-pytorch-master/Facenet_Finetune_Pose_aware_Attention/300W_LPA_profile',
                    help='path to morph')
parser.add_argument('--frontal_folder', type=str,
                    default='/home/moktari/Moktari/2022/facenet-pytorch-master/Pose_Attention_Guided_PIFR/Contrastive_Dataset/cfp-dataset/frontal_test_cropped',
                    help='path to data')
parser.add_argument('--profile_folder', type=str,
                    default='/home/moktari/Moktari/2022/facenet-pytorch-master/Pose_Attention_Guided_PIFR/Contrastive_Dataset/cfp-dataset/profile_test_cropped',
                    help='path to data')
parser.add_argument('--save_folder', type=str,
                    default='/home/moktari/Moktari/2022/facenet-pytorch-master/Pose_Attention_Guided_PIFR/checkpoint_8x8/',
                    help='path to save the data')
args = parser.parse_args()

############################################################
# SET UP Pose Attention-Guided Deep Subspace Learning for PIFR #
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
resnet = FacePoseAwareNet(pose=None)
if torch.cuda.device_count() > 1:  ##  to use both GPUs if available
    print("CHECKING GPUS  AVAILABLE")
    print(torch.cuda.device_count())
    resnet = nn.DataParallel(resnet)
resnet = resnet.to(device)
cudnn.benchmark = True
resnet.train()

####################DataLoader-Initialization#################
train_loader = get_dataset(args)
val_loader = get_dataset_val(args)

####################Hyperparameters-Initialization############
argmargin = 1.4
lr = 0.0001
gamma = 0.1
epochs = 100
patience = 15

##########check parameters required gradient#############
for name, param in resnet.module.named_parameters():
    if param.requires_grad:
        print(name)

optimizer = optim.Adam(resnet.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)

import matplotlib.pyplot as plt


def norm_minmax(x):
    """
    min-max normalization of numpy array
    """
    return (x - x.min()) / (x.max() - x.min())


def plot_tensor(t):
    """
    plot pytorch tensors
    input: list of tensors t
    """
    for i in range(len(t)):
        ti_np = t[i].cpu().detach().numpy().squeeze()
        ti_np = norm_minmax(ti_np)
        if len(ti_np.shape) > 2:
            ti_np = ti_np.transpose(1, 2, 0)
        plt.subplot(1, len(t), i + 1)
        plt.imshow(ti_np)
    plt.show()


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


###############################################################
def validate(epoch):
    resnet.eval()
    loss_m = AverageMeter()
    acc_m = AverageMeter()
    for iter, (img_photo, img_morph, lbl) in enumerate(val_loader):
        bs = img_photo.size(0)
        lbl = lbl.type(torch.float)
        img_photo, img_morph, lbl = img_photo.to(device), img_morph.to(device), lbl.to(device)
        y_photo = resnet(img_photo, pose='frontal')
        y_morph = resnet(img_morph, pose='profile')
        dist = ((y_photo - y_morph) ** 2).sum(1)
        margin = torch.ones_like(dist, device=device) * argmargin
        loss = lbl * dist + (1 - lbl) * F.relu(margin - dist)
        loss = loss.mean()
        acc = (dist < argmargin).type(torch.float)
        acc = (acc == lbl).type(torch.float)
        acc = acc.mean()
        acc_m.update(acc)
        loss_m.update(loss.item())
    print('VALIDATION epoch: %02d, loss: %.4f, acc: %.4f' % (epoch, loss_m.avg, acc_m.avg))
    return loss_m.avg, acc_m.avg


###########################################################################
println = len(train_loader) // 5
chkloss = 10000
step = 0
pl = 0
best_acc = 0
best_epoch = 0
best_all = []
all_step = 0

for epoch in range(epochs):
    print('Ready to train......')
    resnet.train()
    loss_m = AverageMeter()
    acc_m = AverageMeter()
    print('iteration starts...')

    for iter, (img_photo, img_morph, lbl) in enumerate(train_loader):
        bs = img_photo.size(0)
        lbl = lbl.type(torch.float)
        img_photo, img_morph, lbl = img_photo.to(device), img_morph.to(device), lbl.to(device)
        # y_photo = resnet(img_photo)
        # y_morph = resnet(img_morph)
        y_photo = resnet(img_photo, pose='frontal')
        y_morph = resnet(img_morph, pose='profile')
        dist = ((y_photo - y_morph) ** 2).sum(1)
        margin = torch.ones_like(dist, device=device) * argmargin
        loss = lbl * dist + (1 - lbl) * F.relu(margin - dist)
        loss = loss.mean()
        loss.backward()
        optimizer.zero_grad()
        optimizer.step()
        acc = (dist < argmargin).type(torch.float)
        acc = (acc == lbl).type(torch.float)
        acc = acc.mean()
        acc_m.update(acc)
        loss_m.update(loss.item())
        if iter % println == 0:
            print('epoch: %02d, iter: %02d/%02d, loss: %.4f, acc: %.4f' % (
                epoch, iter, len(train_loader), loss_m.avg, acc_m.avg))

    state = {}
    state['resnet'] = resnet.state_dict()
    state['optimizer'] = optimizer.state_dict()
    # torch.save(state,'/home/baaria/Desktop/APPLICATION/CODE/WEIGHTS/SIAMESE/FINE-TUNE'+'BATCH64_LR0.001_MARGIN'+ str(argmargin) + '_model_resnet_'+str(epoch)+'_'+ CURRENT_MORPH+'_VALID.pt')
    val_loss, val_acc = validate(epoch)
    if val_loss > chkloss:
        print("STEP " + str(step + 1) + "\tPLATEAU: " + str(pl) + "\tLR: " + str(lr))
        step += 1
        all_step += 1
        if step > patience:
            best_all.append([best_epoch, chkloss, best_acc, best_weights])
            scheduler.step()
            print("PLATEAU: LOWERING LR...")
            lr = lr * gamma
            #### CONTINUE TRAINING ON LOWER LR FROM THE BEST SAVED WEIGHTS ###
            resnet.load_state_dict(torch.load(best_weights)['resnet'])
            step = 0
            pl += 1
            if all_step > (patience * 2):
                break

    else:
        chkloss = val_loss
        step = 0
        all_step = 0
        best_acc = val_acc
        best_epoch = epoch
        # best_weights = '/home/baaria/Desktop/APPLICATION/CODE/WEIGHTS/SIAMESE/TWINS/IMAGES/RGB/' + 'TWINS_LANDMARK_512_BATCH' +str(args.batch)+'_LR'+ str(lr)+'_MARGIN'+ str(argmargin) + '_model_resnet_' + str(epoch) + '_' + CURRENT_MORPH + '_VALID_BEST.pt'
        best_weights = '/home/moktari/Moktari/2022/facenet-pytorch-master/Pose_Attention_Guided_PIFR/checkpoint_8x8/' + str(
            args.batch_size) + '_LR' + str(lr) + '_MARGIN' + str(argmargin) + '_model_resnet_' + str(
            epoch) + '_VALID_BEST.pt'
        torch.save(state, best_weights)
    print('\n Model Saved! \n')

FINAL_WEIGHTS = '/home/moktari/Moktari/2022/facenet-pytorch-master/Pose_Attention_Guided_PIFR/checkpoint_8x8' + str(
    args.batch_size) + '_LR' + str(lr) + '_MARGIN' + str(argmargin) + '_FINAL_WEIGHTS_VALID' + '.pth'
torch.save(resnet.state_dict(), FINAL_WEIGHTS)
best_all.append([best_epoch, chkloss, best_acc, best_weights])

for best in best_all:
    print(best)
