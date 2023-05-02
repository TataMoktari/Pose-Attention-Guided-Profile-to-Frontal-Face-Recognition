import torch.nn.functional as F
import torch.nn as nn
from cv2 import INTER_LINEAR
from matplotlib import pyplot as plt

from MyModel import *
from nnmodels.inception_resnet_v1 import InceptionResnetV1, InceptionResnetV1_f
import cv2
from MODELS.cbam import CBAM
# from Attention_block import PaB, PaB_P, PaB_Q
from Attention_block import PaB


class FacePoseAwareNet(nn.Module):
    def __init__(self, pose=None):
        super(FacePoseAwareNet, self).__init__()
        self.l0 = msedmodel()
        self.l_a = PaB(512)
        self.lf = resnet_f()
        self.lp = resnet_p()
        self.pose = pose

    def forward(self, x, pose):
        emb = None
        if pose == 'frontal':
            emb = self.lf(x)

        elif pose == 'profile':
            imgs = x.detach().cpu().numpy()[:, :, :, :]
            # x1 = np.resize(imgs, (x.shape[0], x.shape[1], 224, 224))
            # x1 = torch.from_numpy(x1).cuda()
            x1 = torch.from_numpy(imgs).permute(0, 1, 2, 3).float()
            x2 = F.interpolate(x1, size=(224, 224), mode='bilinear')
            x3 = torch.from_numpy(np.asarray(x2)).cuda()
            yaw, yaw_predicted, att_layer4 = self.l0(x3)  ####  when HopeNet resizes 160x160 to 224x224
            sp_matx, ch_matx = self.l_a(att_layer4)
            emb = self.lp(x, ch_matx, sp_matx)

        return emb


class resnet_f(nn.Module):
    def __init__(self):
        super(resnet_f, self).__init__()
        self.inception_resnet_f = InceptionResnetV1_f(classify=False, pretrained='vggface2', num_classes=None,
                                                      num_bins=None)

        layer = 0
        for child in self.inception_resnet_f.children():
            layer += 1
            if layer < 14:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True

    def forward(self, x):
        x = self.inception_resnet_f(x)
        return x


class resnet_p(nn.Module):
    def __init__(self):
        super(resnet_p, self).__init__()
        self.inception_resnet_p = InceptionResnetV1(classify=False, pretrained='vggface2', num_classes=None,
                                                    num_bins=None)

        layer = 0
        for child in self.inception_resnet_p.children():
            layer += 1
            if layer < 14:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True

    def forward(self, x, z, y):
        x = self.inception_resnet_p(x, z, y)
        return x
