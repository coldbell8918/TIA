#
# Authors: Wei-Hong Li
# This code is adapted from https://github.com/srebuffi/residual_adapters

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import models
import os
import time
import argparse
import numpy as np

import datasets
import config_task
import utils_pytorch


parser = argparse.ArgumentParser(description='PyTorch Universal Representation Learning')
parser.add_argument('--dataset', default='cifar100', nargs='+', help='Task(s) to be trained')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--wd', default=1., type=float, help='weight decay for the classification layer')
parser.add_argument('--nb_epochs', default=120, type=int, help='nb epochs')
parser.add_argument('--step1', default=80, type=int, help='nb epochs before first lr decrease')
parser.add_argument('--step2', default=100, type=int, help='nb epochs before second lr decrease')
parser.add_argument('--steps', default=None, type=int, nargs='+', help='nb epochs before second lr decrease')
parser.add_argument('--mode', default='bn', type=str, help='Task adaptation mode')
parser.add_argument('--expdir', default='./results/mdl/', help='Save folder')
parser.add_argument('--datadir', default='./data/decathlon-1.0/', help='folder containing data folder')
parser.add_argument('--imdbdir', default='./data/decathlon-1.0/annotations/', help='annotation folder')
parser.add_argument('--source', default='', type=str, help='Network source')
parser.add_argument('--seed', default=0, type=int, help='seed')
parser.add_argument('--factor', default='1.', type=float, help='Width factor of the network')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--resume-results', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--trainval', dest='trainval', action='store_true', help='using train and validation data')

args = parser.parse_args()
config_task.mode = args.mode
config_task.factor = args.factor
args.use_cuda = torch.cuda.is_available()
if type(args.dataset) is str:
    args.dataset = [args.dataset]

if not os.path.isdir(args.expdir):
    os.mkdir(args.expdir) 
print('=> results and checkpoints will be stored in {}'.format(args.expdir))
args.wd = args.wd *  0.0001

args.ckpdir = args.expdir + '/checkpoint/'
args.svdir  = args.expdir + '/results/'

if not os.path.isdir(args.ckpdir):
    os.mkdir(args.ckpdir) 

if not os.path.isdir(args.svdir):
    os.mkdir(args.svdir) 

#####################################
# Prepare data loaders
train_loaders, val_loaders, num_classes = datasets.prepare_data_loaders(args.dataset,args.datadir,args.imdbdir,True,train_val=args.trainval)
args.num_classes = num_classes
num_classes = {}
for t_index, task in enumerate(args.dataset):
    num_classes[task] = args.num_classes[t_index]

# Create the network
net = models.resnet26(num_classes=list(num_classes.values()))
net.train()
if args.source:
    checkpoint = torch.load(args.source)
    state_dict = {k: v for k, v in checkpoint['net'].items() if 'cls' not in k}
    cls_state_dict = {str.replace(k,'cls_fn','cls_fn.0'): v for k,v in checkpoint['net'].items() if 'cls' in k}
    net.load_state_dict(state_dict, strict=False)
    net.load_state_dict(cls_state_dict, strict=False)

start_epoch = 0
best_acc = 0  # best test accuracy
results = np.zeros((4,start_epoch+args.nb_epochs,len(args.num_classes)))
all_tasks = range(len(args.dataset))
np.random.seed(1993)

if args.use_cuda:
    net.cuda()
    cudnn.benchmark = True


args.criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)

if args.resume:
    print('=> loading checkpoint {} ..'.format(args.resume))
    checkpoint = torch.load(args.resume)
    net.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']+1
    best_acc = checkpoint['acc']
    print('=> checkpoint {} loaded!'.format(args.resume))
    if args.resume_results:
        results = np.load(args.resume_results)
        print('=> results loaded')

print("Start training")
for epoch in range(start_epoch, start_epoch+args.nb_epochs):
    training_tasks = utils_pytorch.adjust_learning_rate_and_learning_taks(optimizer, epoch, args, args.lr)

    st_time = time.time()
    
    # Training and validation
    train_acc, train_loss = utils_pytorch.trainmdl(epoch, train_loaders, training_tasks, net, args, optimizer)
    test_acc, test_loss, best_acc = utils_pytorch.urltest(epoch,val_loaders, all_tasks, net, best_acc, args, optimizer)
        
    # Record statistics
    for i in range(len(training_tasks)):
        current_task = training_tasks[i]
        results[0:2,epoch,current_task] = [train_loss[i].cpu().numpy(),train_acc[i].cpu().numpy()]
    for i in all_tasks:
        results[2:4,epoch,i] = [test_loss[i].cpu().numpy(),test_acc[i].cpu().numpy()]
    np.save(args.svdir+'/results_'+str(args.seed)+args.mode+''.join(args.dataset)+str(args.lr)+str(args.wd)+str(args.nb_epochs)+str(args.step1)+str(args.step2),results)
    print('Epoch lasted {0}'.format(time.time()-st_time))


