import torch
import torch.nn as nn
import hopenet
import torchvision
from MODELS.model_resnet import *
import numpy as np
from torch.autograd import Variable
import pickle

# depth = 50
# att_type = CBAM
#
# class msedmodel(nn.Module):
#     def __init__(self):
#         super(msedmodel, self).__init__()
#         dtype = torch.FloatTensor
#         self.hope_model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66, att_type='CBAM')
#         self.num_of_classes = 13
#         pretrained_dict = torch.load('/home/moktari/Moktari/2022/facenet-pytorch-master/Facenet_Finetune_Pose_aware_Attention/Hopenet_Attention_fine_tune/preprocessed_best_adam/spatial_channel_attention/_epoch_17.pkl')
#         model_dict = self.hope_model.state_dict()
#         # 1. filter out unnecessary keys
#         pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#         # 2. overwrite entries in the existing state dict
#         model_dict.update(pretrained_dict)
#         # 3. load the new state dict
#         self.hope_model.load_state_dict(model_dict)
#
#         # set parameters which needs gradient-decent
#         # layer = 0
#         # for child in self.hope_model.children():
#         #     layer += 1
#         #     if layer < 8:
#         #         for param in child.parameters():
#         #             param.requires_grad = False
#
#         for child in self.hope_model.children():
#             for param in child.parameters():
#                 param.requires_grad = False
#
#         self.hope_model.fc_yaw = nn.Linear(2048, self.num_of_classes)
#         self.idx_tensor = [idx for idx in range(13)]
#         self.idx_tensor = nn.Parameter(torch.FloatTensor(self.idx_tensor), requires_grad=False)
#         self.soft = nn.Softmax()
#
#     def forward(self, x):
#         yaw, pitch, roll, att_layer, att_matrix = self.hope_model(x)
#         yaw_ = self.soft(yaw)
#         yaw_predicted = torch.sum(yaw_ * self.idx_tensor, 1) * 15 - 90
#         return yaw, yaw_predicted, att_layer, att_matrix
#
#
# def msedLoss(output, target):
#     idx_tensor = torch.arange(start=0, end=13).cuda()
#     yaw_predicted = torch.sum(output * idx_tensor, 1) * 3. - 99.
#     loss = torch.mean((yaw_predicted - target) ** 2)
#     return loss

#############updated version which provides feature map of size 2048x7x7############
import torch
import torch.nn as nn
import hopenet
import torchvision
import numpy as np
from torch.autograd import Variable


class msedmodel(nn.Module):
    def __init__(self):
        super(msedmodel, self).__init__()
        dtype = torch.FloatTensor
        self.hope_model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
        pretrained_dict = torch.load('/media/moktari/nlab_Disk/2022/Pose_Aware_facenet/Pose_Attention_Guided_PIFR/FaceNet_PIFR/models/hopenet_robust_alpha1.pkl')
        model_dict = self.hope_model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        self.hope_model.load_state_dict(model_dict)

        # set gradient false
        for param in self.hope_model.parameters():
            param.requires_grad = False

        self.idx_tensor = [idx for idx in range(66)]
        self.idx_tensor = nn.Parameter(torch.FloatTensor(self.idx_tensor), requires_grad=False)
        self.soft = nn.Softmax()

    def forward(self, x):
        yaw, pitch, roll, layer_4 = self.hope_model(x)
        yaw_ = self.soft(yaw)
        yaw_predicted = torch.sum(yaw_ * self.idx_tensor, 1) * 3 - 99
        return yaw, yaw_predicted, layer_4
