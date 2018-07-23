import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from utils import resize_label

root_train = '/home/vrlab/Downloads/pyramidbox/data/WIDER_train/images/'
root_val = '/home/vrlab/Downloads/pyramidbox/data/WIDER_val/images/'



def default_loader(path):
    return Image.open(path).convert('RGB').resize((640, 640))

class MyDataset(Dataset):
    def __init__(self, parse, txt, transform=None, target_transform=None, loader=default_loader):
        if parse == 'train':
            root = root_train
        else:
            root = root_val
        imgs = []
        labels = []
        img_name = ""

        f = open(txt, 'r')
        lines = f.readlines()

        flag = 0
        count = 0
        iter = 0

        for line in lines:
            if flag == 0:
                img_name = root+line.strip()
                flag = 1
                continue
            elif flag == 1:
                count = int(line.strip())
                if count == 0:
                    flag = 0
                else:
                    flag = 2
                continue
            else:
                line = line.strip().split()
                labels.append([float(line[0]), float(line[1]), float(line[2]), float(line[3])])
                iter = iter+1
                if iter == count:
                    imgs.append((img_name, np.array(labels)))
                    iter = 0
                    labels = []
                    flag = 0
                    continue
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        w, h = img.size
        if self.transform is not None:
            img = self.transform(img)
        label = resize_label(label, w, h, 640, 640)
        return img, label

    def __len__(self):
        return len(self.imgs)







