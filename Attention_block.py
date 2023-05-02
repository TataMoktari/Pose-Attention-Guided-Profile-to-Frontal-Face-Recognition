import torch
from torch import Tensor
import torch.nn as nn
# from .._internally_replaced_utils import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List
# from MODELS.cbam import CBAM, CBAM_P, CBAM_Q
from MODELS.cbam import CBAM


class PaB(nn.Module):
    def __init__(self, planes: int) -> None:
        super(PaB, self).__init__()
        self.att_block = CBAM(planes * 4, 16)

    def forward(self, x):
        spa_matrix, cha_matrix = self.att_block(x)
        return spa_matrix, cha_matrix
