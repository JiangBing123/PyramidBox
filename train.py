from __future__ import print_function
import argparse

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms,utils
from torch.utils.data import DataLoader
from dataset import MyDataset
from pyramidbox import build_pyramidbox
from multibox_loss import MultiBoxLoss


root_train = '/home/vrlab/Downloads/pyramidbox/data/WIDER_train/images/'
root_val = '/home/vrlab/Downloads/pyramidbox/data/WIDER_val/images/'





def train(args, model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        target = [torch.Tensor(t).to(device) for t in target]

        data, target = data.to(device), target
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='Weight decay for SGD')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--resume', default=None, type=str,
                        help='Checkpoint state_dict file to resume training from')
    parser.add_argument('--save_folder', default='weights/',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                        help='Pretrained base model')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_data = MyDataset(parse='train', txt=root_train+'train.txt', transform=transforms.ToTensor())
    val_data = MyDataset(parse='val', txt=root_val+'val.txt', transform=transforms.ToTensor())

    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True, **kwargs)
    val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True, **kwargs)

    '''
    #test 
    inputs, labels = next(iter(val_loader))
    '''

    model = build_pyramidbox('train', 640, device).to(device)

    if args.resume:
        print('resume training,loading {}...'.format(args.resume))
        model.load_weights(args.resume)
    else:
        vgg_weights = torch.load(args.save_folder+args.basenet)
        print('loading base network...')
        model.vgg.load_state_dict(vgg_weights)

    if not args.resume:
        print('initializing weights...')
        model.extra_layers.apply(weights_init)
        model.predict_0.apply(weights_init)
        model.predict_1.apply(weights_init)
        model.predict_2.apply(weights_init)
        model.face_head_conf.apply(weights_init)
        # model.head_head_conf.apply(weights_init)
        # model.body_head_conf.apply(weights_init)
        model.face_head_loc.apply(weights_init)
        # model.head_head_loc.apply(weights_init)
        # model.body_head_loc.apply(weights_init)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(overlap_thresh=0.35, num_classes=2, device=device)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, criterion, epoch)
        torch.save(model.state_dict(), 'weights/Pyramidbox_widerface_' +
                   repr(epoch) + '.pth')



def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()

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



if __name__ == '__main__':
    main()