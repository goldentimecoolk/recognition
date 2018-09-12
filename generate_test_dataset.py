import os
import math
import shutil

def sum(path):
    imgnum = 0
    classes = os.listdir(path)
    for classi in classes:
        classdir = os.path.join(path,classi)
        imgs = os.listdir(classdir)
        imgnum += len(imgs)
        #print(imgnum)
    return imgnum

acc = 0.4
dst = '/home/jsk/s/prcv/dataset/v3/test40'
src1 = '/home/jsk/s/prcv/dataset/v2/testresult/right'
src2 = '/home/jsk/s/prcv/dataset/v2/testresult/left'

total = sum(src1)+sum(src2)

def select(src1,src2,dst,total,acc):
    cent = 5000.0/total
    classes = os.listdir(src1)
    for classi in classes:
        ssrc1 = os.path.join(src1,classi)
        ssrc2 = os.path.join(src2,classi)
        ddst = os.path.join(dst,classi)
        if not os.path.exists(ddst):
            os.makedirs(ddst)     #os.mkdir(ddst)
        imgs1 = os.listdir(ssrc1)
        imgs2 = os.listdir(ssrc2)

        i1 = len(imgs1)
        i2 = len(imgs2)
        num_right = math.floor((i1+i2)*cent*acc)
        #avoid overflow
        if num_right>i1:
            num_right = i1
            num_false = i1*(1./acc-1)
        else:
            num_false = math.floor((i1+i2)*cent*(1-acc))
        for ind, img in enumerate(imgs1):
            src = os.path.join(ssrc1,img)
            shutil.copy(src,ddst)
            if ind >= num_right:
                break
        for ind, img in enumerate(imgs2):
            src = os.path.join(ssrc2,img)
            shutil.copy(src,ddst)
            if ind > num_false:
                break

select(src1,src2,dst,total,acc)
print(sum(dst))
