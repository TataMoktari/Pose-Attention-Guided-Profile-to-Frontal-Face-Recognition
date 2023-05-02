import sklearn.metrics as metrics
import argparse
import seaborn as sns
import plotly.offline as py
#################################################################################################
import torch
import plotly.express as px
from scipy.optimize import fsolve
from sklearn.metrics import top_k_accuracy_score
from torch import nn

# import binary_plots

from Facenet_tune import FacePoseAwareNet
from validation_dataset import get_dataset_val
import numpy as np

# sns.set_style("darkgrid")

parser = argparse.ArgumentParser(description='Contrastive view')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--margin', default=0.85, type=int, help='batch size')
parser.add_argument('--frontal_folder', type=str,
                    default='/home/moktari/Moktari/2022/facenet-pytorch-master/Pose_Attention_Guided_PIFR/Contrastive_Dataset/cfp-dataset/frontal_test_cropped',
                    # default='/home/moktari/Moktari/2022/facenet-pytorch-master/Facenet_Finetune_Pose_aware_Attention/Contrastive_Dataset/cfp-dataset/frontal_test_cropped',
                    help='path to data')
parser.add_argument('--profile_folder', type=str,
                    default='/home/moktari/Moktari/2022/facenet-pytorch-master/Pose_Attention_Guided_PIFR/Contrastive_Dataset/cfp-dataset/profile_test_cropped',
                    # default='/home/moktari/Moktari/2022/facenet-pytorch-master/Facenet_Finetune_Pose_aware_Attention/Contrastive_Dataset/cfp-dataset/profile_test_cropped',
                    help='path to morph')

args = parser.parse_args()
device = torch.device('cuda:0')

# SET UP Pose Attention-Guided Deep Subspace Learning for PIFR #
state = torch.load(
    '/home/moktari/Moktari/2022/facenet-pytorch-master/Pose_Attention_Guided_PIFR/checkpoint/32_LR0.0001_MARGIN1.4_model_resnet_24_VALID_BEST.pt')
resnet = FacePoseAwareNet(pose=None)
resnet = torch.nn.DataParallel(resnet)
resnet.to(device)
resnet.load_state_dict(state['resnet'])
resnet.eval()

val_loader = get_dataset_val(args)
print(len(val_loader))


def calc_eer(fpr, tpr, method=0):
    if method == 0:
        min_dis, eer = 100.0, 1.0
        for i in range(fpr.size):
            if fpr[i] + tpr[i] > 1.0:
                break
            mid_res = abs(fpr[i] + tpr[i] - 1.0)
            if mid_res < min_dis:
                min_dis = mid_res
                eer = fpr[i]
        return eer
    else:
        f = lambda x: np.interp(x, fpr, tpr) + x - 1
        return fsolve(f, 0.0)


dist_l = []
lbl_l = []
for iter, (img_photo, img_morph, lbl) in enumerate(val_loader):
    # plot_tensor([img_photo[0], img_print[0]])
    print(iter)
    bs = img_photo.size(0)
    lbl = lbl.type(torch.float)

    img_photo, img_morph, lbl = img_photo.to(device), img_morph.to(device), lbl.to(device)
    y_photo = resnet(img_photo, pose='frontal')
    y_morph = resnet(img_morph, pose='profile')

    dist = ((y_photo - y_morph) ** 2).sum(1)
    dist_l.append(dist.data)
    lbl_l.append((1 - lbl).data)

dist = torch.cat(dist_l, 0)
lbl = torch.cat(lbl_l, 0)
dist = dist.cpu().detach().numpy()
lbl = lbl.cpu().detach().numpy()

# np.save('./data/pr_pr_pix2pix_verifier3_lbl.npy', lbl)
# np.save('./data/pr_pr_pix2pix_verifier3_dist.npy', dist)

fpr, tpr, threshold = metrics.roc_curve(lbl, dist)
roc_auc = metrics.auc(fpr, tpr)
eer = calc_eer(fpr, tpr)
print(eer, 'eer')
accuracy = top_k_accuracy_score(lbl, dist)
print(accuracy, 'accuracy', )
print(np.std(accuracy), 'std_deviation')
import matplotlib.pyplot as plt

plt.style.use('seaborn-darkgrid')
plt.figure(figsize=(6, 6))
plt.title('Receiver Operating Characteristic')
# binary_plots.auc_plot(fpr, tpr, ['Logit'], save_plot='AUC1.png')
# plt.savefig(save_plot, dpi=500, bbox_inches='tight')
plt.plot(fpr, tpr, 'r', label='Test, AUC = %0.6f' % roc_auc)
# sns.lineplot(fpr, tpr, color='r', label='Test, AUC = %0.6f' % roc_auc, ci=None)

# plt.legend(loc='lower right')
# fig.xcale('log')
# fig.update_layout(xaxis_type="log")
# fig.update_layout(xaxis_range=[0.003, 1])
# fig.update(layout_xaxis_range=[0.003, 1])
# fig.rcParams.update({'font.size': 32})
# fig.rc('axes', titlesize=30)
# fig.ylabel('True Positive Rate')
# fig.xlabel('False Positive Rate')
# fig.xlim(0.003, 1)
# fig.ylim([0.0, 1])
# fig.show()
# fig.savefig("ROC.png")

auc = metrics.auc(fpr, tpr)
print('Area Under Curve (AUC): %1.3f' % auc)
# eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
# print('Equal Error Rate (EER): %1.3f' % eer)
plt.grid(True, which="both")
plt.xscale('log')
plt.rcParams.update({'font.size': 32})
plt.rc('axes', titlesize=30)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.xlim(0.003, 1)
plt.ylim([0.0, 1])
# plt.show()
plt.savefig("ROC.png", dpi=500, bbox_inches='tight')

# fig.suptitle('test title', fontsize=20)
# plt.xlabel('xlabel', fontsize=18)
# plt.ylabel('ylabel', fontsize=16)


# plt.plot([0, 1], [0, 1], 'r--')
# plt.figure()
# plt.yticks(np.arange(0.0, 1.05, 0.05))
# plt.xticks(np.arange(0.0, 1.05, 0.05))
