import argparse
import os
import shutil
import time
import numpy as np
import torch
import sys
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torchvision import models
import torchvision.datasets as datasets
from common import *
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
import pandas as pd
import cv2
from PIL import Image

from cbam import *
from bam import *
import math


parser = argparse.ArgumentParser(description='Propert for CIFAR10 in pytorch')


parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('--epochs', default=155, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

parser.add_argument('-b', '--batch-size', default=128, type=int,                    
                    metavar='N', help='mini-batch size (default: 128)')

parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')

parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')

parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')

parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)

parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
best_prec1 = 0


# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
    
normalize = transforms.Normalize(
                         (0.507, 0.487, 0.441), (0.267, 0.256, 0.276)
                     )

    
IMAGE_PATH = './ImageNet100/'
valdata = np.load('./20210812imagenet_val.npy')
traindata = np.load('./20210812imagenet_train.npy')
#valdata = valdata[0:2000]
#traindata = traindata[0:2000]
torch.set_num_threads(20)

def default_loader(path):

    return Image.open(path).convert("RGB")
   # return img

class GetLoader(torch.utils.data.Dataset):
    #def __init__(self,data_root,data_label):
    def __init__(self,file,loader = default_loader):
        
        imgs = []

        for i in range(len(file)):
            imgs.append((IMAGE_PATH + file[i][0] + "/" + file[i][1],int(file[i][2])))
        
        self.imags = imgs
        self.loader = loader
        self.transform = transforms.Compose(
            [transforms.Resize(256),
            #transforms.CenterCrop(224),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(60),
            transforms.ToTensor(),
            transforms.Normalize(
                mean = [0.485,0.456,0.406],
                std = [0.229,0.224,0.225,])
            ]
            
            )
        
    def __getitem__(self,index):
        
        data1,datalabel = self.imags[index]

        data2 = self.loader(data1)
        
        data2 = self.transform(data2)

        labels = datalabel
        
        return data2,labels
    
    def __len__(self):
        return len(self.imags)
    
    

train_dataset = GetLoader(traindata)

train_loader = torch.utils.data.DataLoader(
     train_dataset,
     batch_size=64,
     shuffle=True,
     num_workers=1,
     pin_memory=True,
 )

val_data = GetLoader(valdata)

val_loader = torch.utils.data.DataLoader(
     val_data,
     batch_size=64,
     shuffle=True,
     num_workers=1,
     pin_memory=True,
 )
          
       

def main():
    global args, best_prec1
    args = parser.parse_args()
    list_train = [1,2,3]
    list_val = [1,2,3]
    save_path2 = os.path.join('save_temp', 'model.pth')

    #model = torch.load(save_path2)
    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # save_path2 = os.path.join('save_temp', 'model.pth')

    # model = torch.load(save_path2)
    
    model = models.mobilenet_v2(pretrained=False, num_classes = 100)
    
    #model = CNN()

    print(model)
    input()
    
    model.cuda()
    # print(model)
    # input()
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    if args.half:
        model.half()
        criterion.half()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0 = 5, T_mult = 2, eta_min = 0.0001)



    if args.evaluate:
        validate(val_loader, model, criterion)
        return


    for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                print(m.weight.data)
                break

    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        #print('current lr {:.5e}'.format(args.lr))
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))

        train(train_loader, model, criterion, optimizer, epoch, list_train)
        
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                print(m.weight.data)
                print(m.bias.data)
                break
            
        lr_scheduler.step()

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch, list_val)

        if best_prec1 < prec1:
            best_prec1 = prec1
            
            #print("best_prec:",best_prec1)
            
            save_path = os.path.join(args.save_dir, 'model.pth')
            torch.save(model, save_path)
            
            save_checkpoint({
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, filename=os.path.join(args.save_dir, 'stat_model.th'))
        print("best_prec:",best_prec1)
           
        #torch.save(state, filename)
def train(train_loader, model, criterion, optimizer, epoch,list_train):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        if args.half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)
        
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]

        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        
       
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))
    list_train= [epoch, top1.avg,losses.avg]
    data = pd.DataFrame([list_train])
    data.to_csv('./train.csv',mode = 'a',header = False, index = False)
     

def validate(val_loader, model, criterion, epoch_val, list_val):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            if args.half:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))


           
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))
    
    list_val= [epoch_val, top1.avg,losses.avg]
    data = pd.DataFrame([list_val])
    data.to_csv('./val.csv',mode = 'a',header = False, index = False)
    
        
    
    return top1.avg

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
