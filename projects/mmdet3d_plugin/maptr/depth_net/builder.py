import torch.nn as nn
from mmcv.utils import Registry, build_from_cfg
DEPTHNET = Registry("fusers")
def build_depth_net(cfg):
    return DEPTHNET.build(cfg)