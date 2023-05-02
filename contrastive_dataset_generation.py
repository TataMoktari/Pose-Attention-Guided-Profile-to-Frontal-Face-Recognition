import random
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, utils
import torch
from utils import *
import argparse

# parser = argparse.ArgumentParser(description='Contrastive view')
# parser.add_argument('--batch_size', default=96, type=int, help='batch size')
# parser.add_argument('--photo_folder', type=str,
#                     default='/home/nasser/Moktari/2022/facenet-pytorch-master/Facenet_Finetune_Pose_aware_Attention/Contrastive_Dataset/dataset/frontal/',
#                     help='path to data')
# parser.add_argument('--print_folder', type=str,
#                     default='/home/nasser/Moktari/2022/facenet-pytorch-master/Facenet_Finetune_Pose_aware_Attention/Contrastive_Dataset/dataset/Profile/',
#                     help='path to data')
#
# parser.add_argument('--save_folder', type=str,
#                     default='./checkpoint/',
#                     help='path to save the data')
#
# args = parser.parse_args()

#
# class ImageFolderWithPaths(datasets.ImageFolder):
#     """Custom dataset that includes image file paths. Extends
#     torchvision.datasets.ImageFolder
#     """
#
#     # override the __getitem__ method. this is the method that dataloader calls
#     def __getitem__(self, index):
#         # this is what ImageFolder normally returns
#         original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
#         # the image file path
#         path = self.imgs[index][0]
#         # make a new tuple that includes original and the path
#         tuple_with_path = (original_tuple + (path,))
#         return tuple_with_path
#
#
# class ContrastiveDataset(Dataset):
#     def __init__(self, print_dataset, photo_dataset, positive_prob=0.5):
#         super().__init__()
#         self.print = print_dataset
#         self.photo = photo_dataset
#         self.positive_prob = positive_prob
#         self.h = {}
#
#         print(len(self.print))
#         print(len(self.photo))
#
#         for i in range(len(self.photo)):
#             photo_address = self.photo.imgs[i][0]
#             id = photo_address.Split('/')[-2]
#             for j in range(len(self.print.imgs)):
#                 print_address = self.print.imgs[j][0]
#                 if id in print_address:
#                     if i in self.h:
#                         self.h[i].append(j)
#                     else:
#                         self.h[i] = [j]
#
#
#     def __getitem__(self, index):
#         same_class = random.uniform(0, 1)
#         same_class = same_class > self.positive_prob
#         img_0, label_0 = self.photo[index]
#
#         print_samples = self.h[index]
#         if same_class:
#             rnd_idx = random.randint(0, len(print_samples) - 1)
#             index_1 = print_samples[rnd_idx]
#             img_1, label_1 = self.print[index_1]
#         else:
#             # while True:
#             #     index_1 = random.randint(0, self.__len__() - 1)
#             #     if index_1 not in self.h[index]:
#             #         img_1, label_1 = self.print[index_1]
#             #         break
#
#             index_1 = random.randint(0, self.__len__() - 1)
#             if index_1 not in self.h[index]:
#                 img_1, label_1 = self.print[index_1]
#
#
#
#         # print(same_class, '<<')
#         # plot_tensor([img_0, img_1])
#
#         return img_0, img_1, same_class
#
#     def __len__(self):
#         return min(len(self.print), len(self.photo))

########################################################

class ContrastiveDataset(Dataset):
    def __init__(self, morph_dataset, photo_dataset, positive_prob=0.5):
        super().__init__()
        self.print = morph_dataset
        self.frontal = photo_dataset
        self.positive_prob = positive_prob

        print(len(self.photo)) ### any random folder
        print(len(self.frontal))

        self.positive_h = {}
        self.negative_h = {}

        for i in range(len(self.frontal)):
            # contruct the positive pair correspondence
            img_address = self.frontal.imgs[i][0]
            id = img_address.split('/')[-2]
            if id in self.positive_h:
                self.positive_h[id].append(i)
            else:
                self.positive_h[id] = [i]
            # construct the negative pair correspondence
            for j in range(len(self.photo.imgs)):
                profile_address = self.photo.imgs[j][0]
                if id in profile_address:
                    if id in self.negative_h:
                        self.negative_h[id].append(j)
                    else:
                        self.negative_h[id] = [j]

        # for i in range(len(self.morph)):
        #     # contruct the positive pair correspondence
        #     img_address = self.morph.imgs[i][0]
        #     id = img_address.Split('/')[-2]
        #     if id in self.positive_h:
        #         self.positive_h[id].append(i)
        #     else:
        #         self.positive_h[id] = [i]
        #
        #     # construct the negative pair correspondence
        #     for j in range(len(self.photo.imgs)):
        #         morph_address = self.photo.imgs[j][0]
        #         if id in morph_address:
        #             if id in self.negative_h:
        #                 self.negative_h[id].append(j)
        #             else:
        #                 self.negative_h[id] = [j]

    def __getitem__(self, index):
        same_class = random.uniform(0, 1)
        same_class = same_class > self.positive_prob
        img_0, label_0 = self.frontal[index]

        if same_class: # pick a positive sample
            img_address = self.frontal.imgs[index][0]
            id = img_address.split('/')[-2]
            idx_positive = self.positive_h[id]
            rnd_idx = random.randint(0, len(idx_positive) - 1)
            idx_positive = idx_positive[rnd_idx]
            img_1, label_1 = self.frontal[idx_positive]
        else:
            img_address = self.frontal.imgs[index][0]
            id = img_address.split('/')[-2]
            idx_neg = self.negative_h[id]
            rnd_idx = random.randint(0, len(idx_neg) - 1)
            idx_neg = idx_neg[rnd_idx]
            img_1, label_1 = self.photo[idx_neg]


        # print(same_class, '<<')
        # plot_tensor([img_0, img_1])

        return img_0, img_1, same_class

    def __len__(self):
        # return min(len(self.morph), len(self.photo))
        return len(self.frontal)

def fixed_image_standardization(image_tensor):
    # processed_tensor = (image_tensor - 127.5) / 128.0
    processed_tensor = (image_tensor - .5) / .5
    return processed_tensor

def get_dataset(args):
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    photo_dataset = datasets.ImageFolder(
        args.photo_folder,
        transforms.Compose([
            # transforms.Resize(256),
            # transforms.Pad(16),
            # transforms.RandomCrop(256),
            # transforms.RandomRotation(15),
            # transforms.RandomCrop(256),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            fixed_image_standardization
            # transforms.Normalize(mean=mean, std=std),
        ]))

    print_dataset = datasets.ImageFolder(
        args.print_folder,
        transforms.Compose([
            # transforms.Resize(256),
            # transforms.Pad(16),
            # transforms.RandomCrop(256),
            # transforms.RandomRotation(15),
            # transforms.RandomCrop(256),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            fixed_image_standardization
            # transforms.Normalize(mean=mean, std=std),
        ]))

    train_loader = torch.utils.data.DataLoader(
        ContrastiveDataset(print_dataset, photo_dataset), batch_size=args.batch_size, shuffle=True, pin_memory=True)
    return train_loader



