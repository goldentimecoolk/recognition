import sys

acc1 = 0.0
acc5 = 0.0
correct_1 = 0.0
correct_5 = 0.0
root_dir = '/home/jsk/s/torch/'
filename = 'resnet50_epoch30_ft_batch16.pkl.txt'
filename = root_dir+filename
f = open(filename)
lines = f.readlines()
size = len(lines)
for line in lines:

    item = line.strip('\n').split(' ')
    # in the final evaluation, we should get label from NAME2CLASS.txt
    label = int(item[0].split('/')[-2])
    top1 = int(item[2])
    top5 = [int(i) for i in item[4:]]
    #print(label,top1,top5)
    assert type(label)==type(top1)
    assert top1 == top5[0]
    if label in top5:
        correct_5 += 1
        if label==top1:
            correct_1 += 1
            
acc1 = correct_1/size
acc5 = correct_5/size
print('evaluate on %d images, top1 is %.3f %% top5 is %.3f %%'%(size,acc1*100,acc5*100))
