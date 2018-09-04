import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import pyramidAnchors
import torch.nn.init as init
from detection import *


class Pyramidbox(nn.Module):
    def __init__(self, parse, base, extra_layers, predict_module, head_module, size, device, do_bn=False):
        super(Pyramidbox, self).__init__()
        self.parse = parse
        self.vgg = nn.ModuleList(base)
        self.extra_layers = nn.ModuleList(extra_layers)
        self.predict_0 = nn.ModuleList(predict_module[0])
        self.predict_1 = nn.ModuleList(predict_module[1])
        self.predict_2 = nn.ModuleList(predict_module[2])
        self.face_head_conf = nn.ModuleList(head_module[0])
        self.face_head_loc = nn.ModuleList(head_module[1])
        # self.head_head_conf = nn.ModuleList(head_module[2])
        # self.head_head_loc = nn.ModuleList(head_module[3])
        # self.body_head_conf = nn.ModuleList(head_module[4])
        # self.body_head_loc = nn.ModuleList(head_module[5])
        self.size = size
        self.device = device
        self.priors = pyramidAnchors(self.size)
        self.do_bn = do_bn


        if self.do_bn:
            self.input_bn = nn.BatchNorm2d(3,  momentum=0.5)
            self.bn = nn.BatchNorm2d(512,  momentum=0.5)
            

        if parse=='test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes=1, top_k=200, confidence_thred=0.01, nms_thred=0.45, device=self.device)

    def forward(self, x):
        source = []
        fpns = []
        predict_face_conf = []
        predict_head_conf = []
        predict_body_conf = []
        predict_face_loc = []
        predict_head_loc = []
        predict_body_loc = []

        if self.do_bn:
            x = self.input_bn(x)

        for k in range(16):
            x = self.vgg[k](x)
        source.append(x)                 #conv3_3

        for k in range(16, 23):
            x = self.vgg[k](x)
        source.append(x)                #conv4_3

        for k in range(23, 30):
            x = self.vgg[k](x)
        source.append(x)                #conv5_3

        for k in range(30, len(self.vgg)):
            x = self.vgg[k](x)
        source.append(x)                  #conv_fc7

        c0 = nn.Sequential(nn.Conv2d(source[3].size()[1], source[2].size()[1], kernel_size=1),
                           nn.ConvTranspose2d(source[2].size()[1], source[2].size()[1], 3, stride=2, padding=1,
                                              output_padding=1))

        c1 = nn.Sequential(nn.Conv2d(source[2].size()[1], source[1].size()[1], kernel_size=1),
                           nn.ConvTranspose2d(source[1].size()[1], source[1].size()[1], 3, stride=2, padding=1,
                                              output_padding=1))

        c2 = nn.Sequential(nn.Conv2d(source[1].size()[1], source[0].size()[1], kernel_size=1),
                           nn.ConvTranspose2d(source[0].size()[1], source[0].size()[1], 3, stride=2, padding=1,
                                              output_padding=1))
        c0 = c0.to(self.device)
        c1 = c1.to(self.device)
        c2 = c2.to(self.device)

        c0.apply(weights_init)
        c1.apply(weights_init)
        c2.apply(weights_init)

        fpn0 = c0(source[3])+source[2]
        fpn1 = c1(fpn0)+source[1]
        fpn2 = c2(fpn1)+source[0]

        fpns.append(F.normalize(fpn2)*20)
        fpns.append(F.normalize(fpn1)*20)
        fpns.append(F.normalize(fpn0)*20)
        fpns.append(x)

        for k, v in enumerate(self.extra_layers):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                fpns.append(x)

        if self.do_bn:
            fs = [256, 512, 512, 1024, 512, 256]
            for k, x in enumerate(fpns):
                bn = nn.BatchNorm2d(fs[k],  momentum=0.5).to(self.device)
                fpns[k] = bn(x)

        for (x, p0, p1, p2, h1, h2) in zip(fpns, self.predict_0, self.predict_1, self.predict_2,     # , h3, h4, h5, h6
                                                           self.face_head_conf, self.face_head_loc):  # ,self.head_head_conf, self.head_head_loc,self.body_head_conf, self.body_head_loc
            concat = torch.cat((p0(x), p1(x), p2(x)), 1)
            if self.do_bn:
                concat = self.bn(concat)
            predict_face_conf.append(h1(concat).permute(0, 2, 3, 1).contiguous())
            predict_face_loc.append(h2(concat).permute(0, 2, 3, 1).contiguous())
            # predict_head_conf.append(h3(concat).permute(0, 2, 3, 1).contiguous())
            # predict_head_loc.append(h4(concat).permute(0, 2, 3, 1).contiguous())
            # predict_body_conf.append(h5(concat).permute(0, 2, 3, 1).contiguous())
            # predict_body_loc.append(h6(concat).permute(0, 2, 3, 1).contiguous())
        face_conf = torch.cat([o.view(o.size(0), -1) for o in predict_face_conf], 1)
        # head_conf = torch.cat([o.view(o.size(0), -1) for o in predict_head_conf], 1)
        # body_conf = torch.cat([o.view(o.size(0), -1) for o in predict_body_conf], 1)
        face_loc = torch.cat([o.view(o.size(0), -1) for o in predict_face_loc], 1)
        # head_loc = torch.cat([o.view(o.size(0), -1) for o in predict_head_loc], 1)
        # body_loc = torch.cat([o.view(o.size(0), -1) for o in predict_body_loc], 1)

        if self.parse == 'test':
            output = self.detect(
                face_conf.view(face_conf.size(0), -1, 4),
                face_loc.view(face_loc.size(0), -1, 4)
            )
        else:
            output = (
                face_conf.view(face_conf.size(0), -1, 4),
                face_loc.view(face_loc.size(0), -1, 4),
                # head_conf.view(head_conf.size(0), -1, 2),
                # head_loc.view(head_loc.size(0), -1, 4),
                # body_conf.view(body_conf.size(0), -1, 2),
                # body_loc.view(body_loc.size(0), -1, 4),
                self.priors
            )

        return output


def vgg(cfg, in_channel, batch_norm=False):
    layers = []
    in_channels = in_channel
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def extra_layers(cfg, in_channel):
    layers = []
    in_channels = in_channel
    for k, v in enumerate(cfg):
        if k % 2 == 0:
            layers += [nn.Conv2d(in_channels, v, kernel_size=1)]
        else:
            layers += [nn.Conv2d(in_channels, v, kernel_size=3, stride=2, padding=1)]
        in_channels = v
    return layers


def multibox(vgg, extra_layers, cfg):
    p1 = []
    p2 = []
    p3 = []

    count = 0

    fs = [256, 512, 512, 1024, 512, 256]
    # print(vgg[14].out_channels, vgg[21].out_channels, vgg[28].out_channels)
    for v in fs:
        in_channels = v
        layers1 = nn.Sequential()

        for c in cfg[0:3]:
            if c == 1024:
                layers1.add_module('conv2d'+str(count), nn.Conv2d(in_channels, c, kernel_size=3, padding=1))
            else:
                layers1.add_module('conv2d'+str(count), nn.Conv2d(in_channels, c, kernel_size=1))
            in_channels = c
            count = count+1

        layers2 = nn.Sequential(layers1)

        for c in cfg[3:6]:
            if c == 1024:
                layers2.add_module('conv2d'+str(count), nn.Conv2d(in_channels, c, kernel_size=3, padding=1))
            else:
                layers2.add_module('conv2d'+str(count), nn.Conv2d(in_channels, c, kernel_size=1))
            in_channels = c
            count = count + 1

        layers3 = nn.Sequential(layers2)

        for c in cfg[6:9]:
            if c == 1024:
                layers3.add_module('conv2d'+str(count), nn.Conv2d(in_channels, c, kernel_size=3, padding=1))
            else:
                layers3.add_module('conv2d'+str(count), nn.Conv2d(in_channels, c, kernel_size=1))
            in_channels = c
            count = count + 1

        p1.append(layers1)
        p2.append(layers2)
        p3.append(layers3)

    face_head_conf = []
    face_head_loc = []
    head_head_conf = []
    head_head_loc = []
    body_head_conf = []
    body_head_loc = []

    for i in range(6):
        face_head_conf += [nn.Conv2d(512, 4, kernel_size=3, padding=1)]
        face_head_loc += [nn.Conv2d(512, 4, kernel_size=3, padding=1)]
        head_head_conf += [nn.Conv2d(512, 2, kernel_size=3, padding=1)]
        head_head_loc += [nn.Conv2d(512, 4, kernel_size=3, padding=1)]
        body_head_conf += [nn.Conv2d(512, 2, kernel_size=3, padding=1)]
        body_head_loc += [nn.Conv2d(512, 4, kernel_size=3, padding=1)]

    return vgg, extra_layers, (p1, p2, p3), (face_head_conf, face_head_loc)  # , head_head_conf, head_head_loc, body_head_conf, body_head_loc


base_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
extra_cfg = [256, 512, 128, 256]
mbox_cfg = [1024, 256, 256, 1024, 256, 128, 1024, 256, 128]



def build_pyramidbox(parse, size, device):
    base, extra, predict_module, head_module = multibox(vgg(base_cfg, 3), extra_layers(extra_cfg, 1024), mbox_cfg)

    return Pyramidbox(parse, base, extra, predict_module, head_module, size, device, True)

def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()




