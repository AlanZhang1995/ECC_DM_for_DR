import os
from torch.utils import data
from torchvision import transforms as T 
from PIL import Image
import torch as t 
import csv
from random import shuffle, sample
from numpy.random import choice
import numpy as np 
import cv2
import pandas as pd

class DDR_dataset(data.Dataset):
    def __init__(self, train=True, val=False, test=False, multi=25, fold=None, sync_data=False, ratio=None):
        self.test = test
        self.train = train
        self.val = val
        self.multi = multi
        self.path = '/home/haochen/DataComplex/universe/DataSet/diabet_fundus/DDR-dataset/DR_grading/'
        self.sync_path = '/home/haochen/DataComplex/universe/DataSet/Retina_fundus2/'
        self.imgs = []

        if test:
            with open(self.path+'test.txt',encoding='utf-8') as file:
                for line in file.readlines():
                    line = line.strip('\n')
                    img = line.split(' ')[0]
                    label = int(line.split(' ')[1])
                    if label != 5:
                        #self.imgs.append([self.path+'test_preprocessed1024/'+img, label]) #test_preprocessed1024
                        self.imgs.append([self.path+'all_data_preprocessed1024/'+img, label])
            
        elif val:
            if fold == None:
                val_txt = 'valid.txt'
            else:
                val_txt = 'balanced_txt/valid{}.txt'.format(fold)
            with open(self.path+val_txt,encoding='utf-8') as file:
                for line in file.readlines():
                    line = line.strip('\n')
                    img = line.split(' ')[0]
                    label = int(line.split(' ')[1])
                    if label != 5:
                        self.imgs.append([self.path+'all_data_preprocessed1024/'+img, label])

        elif train:
            if fold == None:
                train_txt = 'train.txt'
            else:
                train_txt = 'balanced_txt/train{}.txt'.format(fold)
            with open(self.path+train_txt,encoding='utf-8') as file:
                for line in file.readlines():
                    line = line.strip('\n')
                    img = line.split(' ')[0]
                    label = int(line.split(' ')[1])
                    if label != 5:
                        self.imgs.append([self.path+'all_data_preprocessed1024/'+img, label])
            print('train #', len(self.imgs))
            if sync_data:
                if sync_data == 'w/o':
                    train_txt = 'sync_wo_selection.txt'
                elif sync_data == 'w/':
                    train_txt = 'sync_w_selection.txt'
                else:
                    assert False, 'Wrong sync_data type: {}'.format(sync_data)
                with open(self.sync_path+train_txt,encoding='utf-8') as file:
                    for line in file.readlines():
                        line = line.strip('\n')
                        img = line.split(' ')[0]
                        label = int(line.split(' ')[1])
                        if label == 1:
                            self.imgs.append([self.sync_path+'mild DR/'+img, label])
                        elif label == 3:
                            self.imgs.append([self.sync_path+'severe DR/'+img, label])
                        else:
                            assert False, 'Werid sample with label={}'.format(label)
                print('train # with sync data', len(self.imgs))



        self.imglen = len(self.imgs)
        print(self.imglen)
        self.nor = T.Normalize( # original
            mean=[.426, .298, .213],std=[.277, .203, .169])

        data_aug = {
            'brightness': 0.4,  # how much to jitter brightness
            'contrast': 0.4,  # How much to jitter contrast
            'saturation': 0.4,
            'hue': 0.1,
            'scale': (0.8, 1.2),  # range of size of the origin size cropped
            'ratio': (0.8, 1.2),  # range of aspect ratio of the origin aspect ratio cropped
            'degrees': (-180, 180),  # range of degrees to select from
            'translate': (0.2, 0.2)  # tuple of maximum absolute fraction for horizontal and vertical translations
        }
        if train:
            self.transform = T.Compose([
                T.Resize((640,640)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                # T.ColorJitter(
                #     brightness=data_aug['brightness'],
                #     contrast=data_aug['contrast'],
                #     saturation=data_aug['saturation'],
                #     hue=data_aug['hue']
                # ),
                T.RandomResizedCrop(
                    size=512, # size=(512, 512),
                    scale=data_aug['scale'],
                    ratio=data_aug['ratio']
                ),
                T.RandomAffine(
                    degrees=data_aug['degrees'],
                    # translate=data_aug['translate']
                ),
                # T.RandomGrayscale(0.2),
                T.ToTensor(),
                self.nor
            ])

        elif val or test:
            self.transform = T.Compose([
                T.Resize((512,512)),
                T.ToTensor(),
                self.nor
            ])


    def __getitem__(self, index):
        img, label_grad = self.imgs[index]
        data = Image.open(img).convert('RGB')
        data = self.transform(data)

        label_clf = 0 if label_grad == 0 else 1

        if self.multi == 25: #cam
            # return data, label_clf, label_grad, img  # cam
            return data, label_clf, label_grad  
        elif self.multi == 2:
            return data, label_clf
        elif self.multi == 5:
            return data, label_grad
        


    def __len__(self):
        return len(self.imgs)




if __name__ == '__main__':
    dst = DDR_dataset(train=True,val=False,test=False,multi=5)
    for index in range(dst.__len__()):
        data, label_grad = dst.__getitem__(index)
        print(label_grad)