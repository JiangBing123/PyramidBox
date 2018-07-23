import torch
import numpy as np
import torch.nn.functional as F

def resize_label(labels, w1, h1, w2, h2):
    labs = []
    for label in labels:
        labs.append(np.array([label[0]*w2/w1, label[1]*h2/h1, label[2]*w2/w1, label[3]*h2/h1]))
    return labs



def pyramidAnchors(size):
    priors = []

    stride = [4, 8, 16, 32, 64, 128]
    anchor = [16, 32, 64, 128, 256, 512]

    for k in range(len(stride)):
        fs = int(size/stride[k])
        prior = []
        for i in range(fs):
            for j in range(fs):
                x = float(stride[k]*(i+0.5))
                y = float(stride[k]*(j+0.5))
                w = float(anchor[k])
                h = float(anchor[k])
                prior.append([x, y, w, h])
        priors.append(prior)

    return priors


def rescale_targets(spa, k, targets):
    targets[:, 2:] = targets[:, 2:]/(spa**k)


def point_form(boxes):
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax


def center_size(boxes):
    return torch.cat((boxes[:, 2:] + boxes[:, :2])/2,  # cx, cy
                     boxes[:, 2:] - boxes[:, :2], 1)  # w, h


def intersect(box_a, box_b):
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def match(spa, k, anchor, threshold, targets, priors, conf_t, loc_t, idx, device):
    rescale_targets(spa, k, targets)

    ps_conf = [0]*len(priors)
    ps_loc = [0]*len(priors)

    for i in range(len(priors)):
        ps_conf[i] = torch.Tensor(len(priors[i])).zero_().to(device)
        ps_loc[i] = torch.Tensor(len(priors[i]), 4).zero_().to(device)

    ind = 0
    dist = float('inf')

    for target in targets:
        for i in range(len(anchor)):
            if (abs((0.5*target[2]+0.5*target[3])-anchor[i])) < dist:
                dist = abs((0.5*target[2]+0.5*target[3])-anchor[i])
                ind = i
        dist = float('inf')

        overlaps = jaccard(point_form(target.unsqueeze(0)),
                           point_form(torch.Tensor(priors[ind]).to(device)))    # [1,k]
        overlaps = overlaps.gt(threshold).squeeze(0).float()

        ps_conf[ind] = ((ps_conf[ind]+overlaps).gt(0)).float()


        indice = overlaps.nonzero()

        if indice.size(0) == 0:
            continue

        cord = target.unsqueeze(0).expand(indice.size(0), 4)
        ps_loc[ind].index_copy_(0, indice.squeeze(1), cord)  # not correct

    conf = ps_conf[0]
    matched = ps_loc[0]
    prs = torch.Tensor(priors[0]).to(device)

    for i in range(1, len(priors)):
        conf = torch.cat((conf, ps_conf[i]), 0)
        matched = torch.cat((matched, ps_loc[i]), 0)
        prs = torch.cat((prs, torch.Tensor(priors[i]).to(device)), 0)

    loc = encode(matched, prs)

    conf_t[idx] = conf
    loc_t[idx] = loc


def encode(matched, priors):

    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] - priors[:, :2])/priors[:, :2]

    # match wh / prior wh
    g_wh = torch.log(matched[:, 2:] / priors[:, 2:])

    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]


def decode(loc, priors):

    boxes = torch.cat((loc[:, :2]*priors[:, :2]+priors[:, :2],
                       torch.exp(loc[:, 2:])*priors[:, 2:]), 1)

    return boxes


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.max()

    x = torch.log(torch.sum(torch.exp(x-x_max))) + x_max
    #x = F.log_softmax(x, dim=1)

    return x

























