# Adapted from https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from models.out_fns import get_outfns

class SingleTaskModel(nn.Module):
    """ Single-task baseline model with encoder + decoder """
    def __init__(self, backbone: nn.Module, decoder: nn.Module, task: str):
        super(SingleTaskModel, self).__init__()
        self.backbone = backbone
        self.decoder = decoder 
        self.task = task
        self.outfns = get_outfns([task])

    def forward(self, x, return_mid=False):
        out_size = x.size()[2:]
        feats = self.backbone(x)
        if isinstance(feats, list):
            feats = feats[-1]

        # decoder의 출력이 튜플이면 중간 feature도 받기
        out = self.decoder(feats)
        if isinstance(out, tuple):
            out, mid_feat = out
        else:
            mid_feat = None

        # interpolate (decoder가 이미 원해상도인 경우 생략)
        if out.size()[2:] != out_size:
            out = F.interpolate(out, out_size, mode='bilinear', align_corners=True)

        output = {self.task: self.outfns[self.task](out)}

        # 중간 feature가 필요한 경우 함께 반환
        if return_mid:
            return output, mid_feat
        else:
            return output

    def embed(self, x):
        return self.backbone(x)


class MultiTaskModel(nn.Module):
    """ Multi-task baseline model with shared encoder + task-specific decoders """
    def __init__(self, backbone: nn.Module, decoders: nn.ModuleDict, tasks: list):
        super(MultiTaskModel, self).__init__()
        assert(set(decoders.keys()) == set(tasks))
        self.backbone = backbone
        self.decoders = decoders
        self.tasks = tasks
        self.outfns = get_outfns(tasks)

    def forward(self, x, feat=False, return_mid=False):
        out_size = x.size()[2:]
        shared_representation = self.backbone(x)
        feats = shared_representation

        if isinstance(shared_representation, list):
            feats = shared_representation
            shared_representation = shared_representation[-1]

        outputs = {}
        mid_feats = {} if return_mid else None

        for task in self.tasks:
            out = self.decoders[task](shared_representation)

            # 중간 feature 분리
            if isinstance(out, tuple):
                out, mid_feat = out
            else:
                mid_feat = None

            # interpolation 처리
            if out.size()[2:] != out_size:
                out = F.interpolate(out, out_size, mode='bilinear', align_corners=True)

            outputs[task] = self.outfns[task](out)
            if return_mid:
                mid_feats[task] = mid_feat

        if feat and return_mid:
            return outputs, feats, mid_feats
        elif feat:
            return outputs, feats
        elif return_mid:
            return outputs, mid_feats
        else:
            return outputs
