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

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

plt.ion()   # interactive mode

data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

data_dir = '/home/jsk/s/prcv/dataset/v2'
image_datasets = datasets.ImageFolder(os.path.join(data_dir, 'rest_test'), data_transforms)
testloader = torch.utils.data.DataLoader(image_datasets, batch_size=512, shuffle=False, num_workers=32)
              
dataset_sizes = len(image_datasets)
class_names = image_datasets.classes

device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

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

root = '/home/jsk/s/torch/params'

model_dir = 'resnet101'
param_str = '101_2'

model_test = model[model_dir]
num_ftrs = model_test.fc.in_features
model_test.fc = nn.Linear(num_ftrs, 205)
print(model_test)
print('test param %s of model %s' % (param[param_str],model_dir))

param_dir = os.path.join(root,param[param_str])
model_test.load_state_dict(torch.load(param_dir))
model_test = model_test.to(device)
model_test.eval()


correct = 0
total = 0
batch = 0
with torch.no_grad():
    for inputs, labels in testloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model_test(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        batch += 1
        print('batch %d accuracy: %.3f %%' % (batch,100.*correct/total))
print('Accuracy of the %s on the %d test images: %.3f %%' % (param[param_str],total,100 * correct / total))



