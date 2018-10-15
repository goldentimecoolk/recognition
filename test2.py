##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Sky.J.
## Center for Research on Intelligent System and Engineering (RISE). CASIA
## Email: jsktt01@gmail.com
## Time: 2018/10/10
##
## Instruction:
## Evaluation code in this version is used to test datasets directly 
## including 5000 images, without subfolders indicating class/label. 
## Corresponding new method for fetching the dataset is implement.
##
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.autograd import Variable
import shutil

from PIL import Image

root = '/home/jsk/s/torch/params'
num_classes = 205
model_str = 'densenet161'
LEVEL = ['v3_8','16_bn','101_8','161_8', '201_8']
param_str = LEVEL[3]
TEST_OR_COPY = 'TEST'


def default_loader(path):
    try:
        img = Image.open(path)
        return img.convert('RGB')
    except:
        print("Cannot read image: {}".format(path))

class customData(Dataset):
    def __init__(self, img_path, txt_path, dataset = '', data_transforms=None, loader = default_loader):
        with open(txt_path) as input_file:
            lines = input_file.readlines()
            self.img_name = [os.path.join(img_path, line.strip().split()[0]) for line in lines]
            self.img_label = [int(line.strip().split()[1]) for line in lines]
            #print(self.img_name)  # for debug
            #l=(line.strip().split()[0] for line in lines)
            #print(l)  # for debug
            #l.append(line.strip().split()[0]) for line in lines)
            #print(self.img_label)
        self.data_transforms = data_transforms
        self.dataset = dataset
        self.loader = loader

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        img_name = self.img_name[item]
        label = self.img_label[item]
        img = self.loader(img_name)

        if self.data_transforms is not None:
            try:
                img = self.data_transforms(img)
            except:
                print("Cannot transform image: {}".format(img_name))
        return img, label


def main():

    data_root = '/home/jsk/s/prcv/dataset'
    data_dir = os.path.join(data_root,'v2')
    dataset_dir = '/home/jsk/s/prcv/dataset/v4/dataset'
    gt_path = './txt/clean_dataset_v2_5010_50.txt'
    
    pathlist = []

    indexes = os.listdir(dataset_dir)
    indexes.sort()
    for index in indexes:
        dire = os.path.join(dataset_dir,index)
        pathlist.append(dire)
    print('total image is %d'%(len(pathlist)))
    #print(pathlist[394:405])

    if model_str[:9] == 'inception':
        size1 = 299
        size2 = 299
    else:
        size1 = 256
        size2 = 224
    data_transforms = transforms.Compose([
            transforms.Resize(size1),
            transforms.CenterCrop(size2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    image_datasets = customData(img_path=dataset_dir, txt_path=(gt_path), data_transforms=data_transforms) 
    testloader = torch.utils.data.DataLoader(image_datasets, batch_size=1, shuffle=False, num_workers=32)
              
    dataset_sizes = len(image_datasets)
    #assert dataset_sizes==len(pathlist),'pytorch dataset is not equal to pathlist'

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    model = {#'resnet18':models.resnet18(pretrained=False), 
              'resnet50':models.resnet50(pretrained=False), 
              'resnet101':models.resnet101(pretrained=False), 
              #'resnet152':models.resnet152(pretrained=False),
              #'densenet121':models.densenet121(pretrained=False),
              'densenet161':models.densenet161(pretrained=False),
              #'densenet169':models.densenet169(pretrained=False),
              'densenet201':models.densenet201(pretrained=False),
              'inception_v3':models.inception_v3(pretrained=False),
              'vgg16_bn':models.vgg16_bn(pretrained=False),
              'vgg19_bn':models.vgg19_bn(pretrained=False)}

    param = {'v3_8':'inception_v3_epoch30_all_batch8_SGD_0.001.pkl',
             '16_bn':'vgg16_bn_epoch30_all_batch8_SGD_0.001.pkl',
             '101_8':'resnet101_epoch26_ft_batch8.pkl',
             '161_8':'densenet161_epoch30_all_batch8_SGD_0.001.pkl',
             '201_8':'densenet201_epoch30_all_batch8_SGD_0.002.pkl',

             '50_8_sgd':'resnet50_epoch30_ft_batch8.pkl',
             '50_8_adam':'resnet50_epoch30_all_batch8_Adam_6e-05.pkl',
             '101_4':'resnet101_epoch30_all_batch4_SGD_0.0008.pkl',
             'v3_4':'inception_v3_epoch30_all_batch4_SGD_0.001.pkl',
             '161_4':'densenet161_epoch30_all_batch4_SGD_0.0008.pkl'}
    
    model_test = model[model_str]

    if model_str[:6] == 'resnet':
        num_ftrs = model_test.fc.in_features
        model_test.fc = nn.Linear(num_ftrs, num_classes)
    elif model_str[:8] == 'densenet':
        num_ftrs = model_test.classifier.in_features
        model_test.classifier = nn.Linear(num_ftrs, num_classes)
    elif model_str[:9] == 'inception':
        num_ftrs = model_test.fc.in_features
        model_test.fc = nn.Linear(num_ftrs, num_classes)
    elif model_str[:3] =='vgg':
        num_ftrs = model_test.classifier[6].in_features
        model_test.classifier[6] = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError('choose the available')

    print(model_test)
    print('current mode: %s'%(TEST_OR_COPY))
    print('test param %s of model %s' % (param[param_str],model_str))

    param_dir = os.path.join(root,param[param_str])
    model_test.load_state_dict(torch.load(param_dir))
    model_test = model_test.to(device)
    model_test.eval()

    correct = 0
    correct_1 = 0
    correct_5 = 0
    top1 = 0
    top5 = 0
    total = 0
    batch = 0
    num1 = 0
    num2 = 0


    filename = param[param_str]+'.txt'
    file_txt = open(filename, 'w')
    with torch.no_grad():
        index = 0
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model_test(inputs)

            #_, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            #correct += (predicted == labels).sum().item()
            #print(type(inputs))
            assert inputs.size()[0]==1
            img = pathlist[index]
            index += 1
            itop1,itop5 = stand_output(outputs,topk=(1,5))
            #top1_list.append(itop1)
            #top5_list.append(itop5)
            print(img.split('/')[-1]+' top1: '+str(itop1)+' top5: '+" ".join(str(x) for x in itop5))
            str1 = img.split('/')[-1]+' top1: '+str(itop1)+' top5: '+" ".join(str(x) for x in itop5)+'\n'
            file_txt.writelines(str1)
            
            dcorrect_1, dcorrect_5 = accuracy(outputs,labels,topk=(1,5))
            
            correct_1 += dcorrect_1
            correct_5 += dcorrect_5
            top1 = correct_1.float()/total
            top5 = correct_5.float()/total

            batch += 1
            print('batch %d  label %d  correct %d' % (batch,labels.item(),dcorrect_1.item()))
            #print('batch %d accuracy: %.3f %%' % (batch,100.*correct/total))
            #print('batch %d top1 accuracy: %.3f %% top5 accuracy: %.3f %%' % (batch,100*top1,100*top5))
    print('Accuracy of the %s on the %d test images: top1 %.3f %%  top5 %.3f %%' % (param[param_str],total,100*top1,100*top5))
    file_txt.close()


def stand_output(output, topk=(1,)):
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    #print(pred.item())
    pred = torch.squeeze(pred)
    res1 = pred.cpu().numpy()
    res0 = res1[0]
    #print(res)
    return res0, res1


def accuracy(output, target, topk=(1,)):
    
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        #res.append(correct_k.mul_(100.0 / batch_size))
        res.append(correct_k)
    return res

if __name__ == '__main__':
    main()
