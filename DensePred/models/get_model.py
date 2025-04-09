#
# Authors: Wei-Hong Li

import os
import copy
import torch
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.data import DataLoader

def get_model(args, tasks_outputs, teacher=False):
    # Return multi-task learning model or single-task model
    if args.backbone == 'segnet':
        from models.segnet import SegNet
        backbone = SegNet()
        backbone_channels = 64
    elif teacher:
        from models.resnet import resnet152
        backbone = resnet152(pretrained=True)  # 사전 학습된 모델 사용
        backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
        backbone_channels = 2048  # ResNet-50의 마지막 feature map 채널 수
    elif args.backbone == 'resnet50':
        from models.resnet import resnet50
        backbone = resnet50(pretrained=True)  # 사전 학습된 모델 사용
        backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
        backbone_channels = 2048  # ResNet-50의 마지막 feature map 채널 수
    elif args.backbone == 'resnet101':
        from models.resnet import resnet101
        backbone = resnet101(pretrained=True)  # 사전 학습된 모델 사용
        backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
        backbone_channels = 2048  # ResNet-50의 마지막 feature map 채널 수
    elif args.backbone == 'resnet152':
        from models.resnet import resnet152
        backbone = resnet152(pretrained=True)  # 사전 학습된 모델 사용
        backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
        backbone_channels = 2048  # ResNet-50의 마지막 feature map 채널 수
    

    if args.method == 'single-task':
        from models.models import SingleTaskModel
        task = args.task
        head = get_head(args.head, backbone_channels, tasks_outputs[task])
        model = SingleTaskModel(backbone, head, task)
    elif args.method == 'vanilla':
        selected_tasks_outputs = {
            task: task_output for task, task_output in tasks_outputs.items() if task in args.tasks
        }
        from models.models import MultiTaskModel
        heads = torch.nn.ModuleDict({
            task: get_head(args.head, backbone_channels, task_output)
            for task, task_output in selected_tasks_outputs.items()
        })
        model = MultiTaskModel(backbone, heads, args.tasks)

    return model

def get_stl_model(args, tasks_outputs, task):
    # Return single-task learning models
    backbone_name = args.backbone
    if backbone_name == 'segnet':
        from models.segnet import SegNet
        backbone = SegNet()
        backbone_channels = 64
    elif backbone_name == 'resnet50':
        from models.resnet import resnet50
        backbone = resnet50(pretrained=True)  # 사전 학습된 ResNet-50 사용
        backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
        backbone_channels = 2048  # ResNet-50의 최종 피처 맵 채널 수

    from models.models import SingleTaskModel
    head = get_head(args.head, backbone_channels, tasks_outputs[task])
    model = SingleTaskModel(backbone, head, task)
    
    return model

def get_head(head, backbone_channels, task_output):
    """ Return the decoder head """
    if head == 'segnet_head':
        from models.segnet import SegNet_head
        return SegNet_head(backbone_channels, task_output)
    elif head == 'deeplab_head':
        from models.aspp import DeepLabHead
        return DeepLabHead(backbone_channels, task_output)



