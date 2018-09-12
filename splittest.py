from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.autograd import Variable
import shutil

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,6,7"

root = '/home/jsk/s/torch/params'
model_dir = 'resnet50'
param_str = '50_4'
TEST_OR_COPY = 'TEST'

def main():

    data_dir = '/home/jsk/s/prcv/dataset/v2'
    dataset_dir = os.path.join(data_dir,'rest_test')
    right = os.path.join(data_dir,'testresult_52/right')
    left = os.path.join(data_dir,'testresult_52/left')   # not '/testresult/left'

    pathlist = []
    classes = os.listdir(dataset_dir)
    #classes = os.listdir(left)
    classes.sort()
    for classi in classes:
        dire = os.path.join(dataset_dir,classi)
        imgs = os.listdir(dire)
        imgs.sort()
        for img in imgs:
            pathlist.append(os.path.join(dire,img))
    #print(pathlist[394:405])
    data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    
    #image_datasets = datasets.ImageFolder(dataset_dir, data_transforms)
    image_datasets = datasets.ImageFolder(right, data_transforms)
    testloader = torch.utils.data.DataLoader(image_datasets, batch_size=1, shuffle=False, num_workers=32)
              
    dataset_sizes = len(image_datasets)
    #assert dataset_sizes==len(pathlist),'pytorch dataset is not equal to pathlist'
    class_names = image_datasets.classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = {'resnet50':models.resnet50(pretrained=False), 
	    'resnet101':models.resnet101(pretrained=False), 
	    'resnet152':models.resnet152(pretrained=False)}

    param = {'101_64':'resnet101_epoch20_ft_batch64.pkl',
	'101_32':'resnet101_epoch20_ft_batch32.pkl',
	'101_16':'resnet101_epoch26_ft_batch16.pkl',
	'101_8':'resnet101_epoch26_ft_batch8.pkl',
	'101_4':'resnet101_epoch26_ft_batch4.pkl',
	'101_2':'resnet101_epoch26_ft_batch2.pkl',
	'50_128':'resnet50_epoch30_ft_batch128.pkl',  
	'50_64':'resnet50_epoch30_ft_batch64.pkl',
	'50_32':'resnet50_epoch30_ft_batch32.pkl',
	'50_16':'resnet50_epoch30_ft_batch16.pkl',
	'50_8':'resnet50_epoch30_ft_batch8.pkl',
	'50_4':'resnet50_epoch30_ft_batch4.pkl',
	'50_2':'resnet50_epoch30_ft_batch2.pkl',
	'50_1':'resnet50_epoch30_ft_batch1.pkl'}

    model_test = model[model_dir]
    num_ftrs = model_test.fc.in_features
    model_test.fc = nn.Linear(num_ftrs, 205)
    #print(model_test)
    print('test param %s of model %s' % (param[param_str],model_dir))

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
    leftpath = []
    num1 = 0
    num2 = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model_test(inputs)

            #_, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            #correct += (predicted == labels).sum().item()
        
            dcorrect_1, dcorrect_5 = accuracy(outputs,labels,topk=(1,5))
            #print('correct',dcorrect_1)
            correct_1 += dcorrect_1
            correct_5 += dcorrect_5
            top1 = correct_1.float()/total
            top5 = correct_5.float()/total
            #print(str(labels.item()))

            left_dir = os.path.join(left,standarlabel(str(labels.item())))
            right_dir = os.path.join(right,standarlabel(str(labels.item())))
            guara_path(left_dir)
            guara_path(right_dir)


            result = dcorrect_1.item()

            if result == 1:
                num1 += 1
                if TEST_OR_COPY == 'COPY':
                    print(pathlist[batch])
                    print(right_dir)
                    shutil.copy(pathlist[batch],right_dir)
            elif result == 0:
                num2 += 1
                #print('correct')
                #leftpath.append(pathlist[batch])
                if TEST_OR_COPY == 'COPY':
                    print(pathlist[batch])
                    print(left_dir)
                    shutil.copy(pathlist[batch],left_dir)
            else:
                raise Exception("disrupted image")   
            
            batch += 1
            print('batch %d  class %d  result %d'%(batch,labels.item(),result))
            result = 2
            #print('batch %d accuracy: %.3f %%' % (batch,100.*correct/total))
            #print('batch %d top1 accuracy: %.3f %% top5 accuracy: %.3f %%' % (batch,100*top1,100*top5))
    print('class A: %d  class B: %d'%(num1,num2))
    print('Accuracy of the %s on the %d test images: top1 %.3f %%  top5 %.3f %%' % (param[param_str],total,100*top1,100*top5))
    print(leftpath)

def guara_path(path):
    if not os.path.exists(path):
        #os.mkdir(path)
        os.makedirs(path)
        print(path)

def standarlabel(string):
    assert len(string)<=3 & len(string)>=0, 'label outsize'
    if len(string) == 3:
        pass
    elif len(string) ==2:
        string = '0'+string
    elif len(string) ==1:
        string = '00'+string
    return string

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
