from __future__ import print_function
import argparse
import cv2

import os
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms,utils
from torch.utils.data import DataLoader
from dataset import MyDataset
from pyramidbox import build_pyramidbox
from PIL import Image

root_val = '/home/vrlab/Downloads/pyramidbox/data/WIDER_val/images/'



def tst_net(save_folder, net, device, transform=None, thresh=0.9):
    testimgs = []
    testset = 'demo/test.txt'
    f = open(testset)
    lines = f.readlines()
    for line in lines:
        testimgs.append(line.strip('\n'))

    for i in range(len(testimgs)):
        print(testimgs[i])
        img = Image.open(testimgs[i]).convert('RGB').resize((640, 640))
        x = transform(img).unsqueeze(0).to(device)

        detections = net(x)
        j= 0
        image = cv2.imread(testimgs[i])
        image = cv2.resize(image,(640, 640))

        while detections[0, j, 0] > thresh:
            score = detections[0,j,0]
            print(score)
            pt = detections[0, j, 1:].cpu().detach().numpy()
            print(pt)
            coords = (int(pt[0]), int(pt[1]), int(pt[2]), int(pt[3]))
            cv2.rectangle(image, (coords[0], coords[1]), (coords[2], coords[3]), (0,255,0), 4)
            j = j+1
            if j==detections.size()[1]:
                break
        cv2.imwrite('demo/00' + str(i) + '_new.jpg', image)

def main():

    # testing settings
    parser = argparse.ArgumentParser(description='PyramidBox Face Detection')
    parser.add_argument('--trained_model', default='weights/Pyramidbox_widerface_4.pth',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--save_folder', default='eval/', type=str,
                        help='Dir to save results')
    parser.add_argument('--visual_threshold', default=0.6, type=float,
                        help='Final confidence threshold')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    net = build_pyramidbox('test', 640, device)  # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net = net.to(device)
    net.eval()
    print('Finished loading model!')

    '''
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    val_data = MyDataset(parse='val', txt=root_val + 'val.txt', transform=transforms.ToTensor())
    val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True, **kwargs)
    '''

    tst_net(save_folder=args.save_folder, net=net, device=device, transform=transforms.ToTensor(), thresh=args.visual_threshold)

if __name__ == '__main__':
    main()


def collate_fn(batch):
    # Note that batch is a list
    batch = list(map(list, zip(*batch)))  # transpose list of list
    out = None
    # You should know that batch[0] is a fixed-size tensor since you're using your customized Dataset
    # reshape batch[0] as (N, H, W)
    # batch[1] contains tensors of different sizes; just let it be a list.
    # If your num_workers in DataLoader is bigger than 0
    #     numel = sum([x.numel() for x in batch[0]])
    #     storage = batch[0][0].storage()._new_shared(numel)
    #     out = batch[0][0].new(storage)
    batch[0] = torch.stack(batch[0], 0, out=out)
    return batch