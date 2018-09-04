import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import rescale_targets, match, log_sum_exp, match_temp



class MultiBoxLoss(nn.Module):
    def __init__(self, overlap_thresh, num_classes, device):
        super(MultiBoxLoss, self).__init__()
        self.threshold = overlap_thresh
        self.device = device
        self.num_classes = num_classes
        self.spa = 2
        self.anchor = [16, 32, 64, 128, 256, 512]
        self.negpos_ratio = 3

    def forward(self, predictions, targets):
        face_data_conf, face_locdata, priors = predictions   #  head_confdata, head_locdata, body_confdata, body_locdata,
        if np.any(np.isnan(face_data_conf.cpu().detach().numpy())):
            print("detection is nan")

        face_confdata_0, _ = torch.max(face_data_conf[:, :, 0:3], dim=2, keepdim=True)
        face_confdata_1 = face_data_conf[:, :, 3:4]
        face_confdata = torch.cat((face_confdata_0, face_confdata_1), dim=2)


        num = face_data_conf.size(0)
        num_priors = face_data_conf.size(1)

        # Match priors(default boxes) and ground truth
        face_conf_t, face_loc_t = self.match_priors(0, num, num_priors, targets, priors)
        # head_conf_t, head_loc_t = self.match_priors(1, num, num_priors, targets, priors)
        # body_conf_t, body_loc_t = self.match_priors(2, num, num_priors, targets, priors)

        face_pos = face_conf_t > 0
        face_neg = face_conf_t <= 0
        face_posnum = face_pos.sum(dim=1, keepdim=True)

        # head_pos = head_conf_t > 0
        # head_neg = head_conf_t <= 0
        # head_posnum = head_pos.sum(dim=1, keepdim=True)
        #
        # body_pos = body_conf_t > 0
        # body_neg = body_conf_t <= 0
        # body_posnum = body_pos.sum(dim=1, keepdim=True)

        # Localization Loss  (Smooth L1)
        face_loss_l = loc_loss(face_pos, face_locdata, face_loc_t)
        # head_loss_l = loc_loss(head_pos, head_locdata, head_loc_t)
        # body_loss_l = loc_loss(body_pos, body_locdata, body_loc_t)


        # Hard example mining
        face_neg = self.hard_mining(face_confdata, face_conf_t, face_pos, num)
        # head_neg = self.hard_mining(head_confdata, head_conf_t, head_pos, num)
        # body_neg = self.hard_mining(body_confdata, body_conf_t, body_pos, num)

        # Confidence Loss for both pos and neg examples
        face_loss_c = self.conf_loss(face_pos, face_neg, face_confdata, face_conf_t)
        # head_loss_c = self.conf_loss(head_pos, head_neg, head_confdata, head_conf_t)
        # body_loss_c = self.conf_loss(body_pos, body_neg, body_confdata, body_conf_t)


        # Sum the loss
        N_face = face_posnum.sum().item()

        # N_head = head_posnum.sum().item()
        # N_body = body_posnum.sum().item()


        fface = min(1, N_face)
        # fhead = min(1, N_head)
        # fbody = min(1, N_body)

        e = 1e-12

        loss_l = (fface*face_loss_l)/(N_face+e)  # (fface*face_loss_l)/(N_face+e) + (fhead*head_loss_l)/(N_head+e) + (fbody*body_loss_l)/(N_body+e)
        loss_c = (fface*face_loss_c)/(N_face+e)  # (fface*face_loss_c)/(N_face+e) + (fhead*head_loss_c)/(N_head+e) + (fbody*body_loss_c)/(N_body+e)

        print(loss_l, loss_c)

        return loss_c + loss_l

    def match_priors(self, k, num, num_priors, targets, priors):
        conf_t = torch.Tensor(num, num_priors)
        loc_t = torch.Tensor(num, num_priors, 4)

        # rescale_targets(self.spa, k, targets)

        for idx in range(num):
            truths = targets[idx][:, :-1]
            labels = targets[idx][:, -1]

            match_temp(self.threshold, truths, priors, labels, loc_t, conf_t, idx, self.device)
            # match(self.spa, k, self.anchor, self.threshold, targets[idx], priors, conf_t, loc_t, idx, self.device)

        conf_t = conf_t.to(self.device)
        loc_t = loc_t.to(self.device)

        return conf_t, loc_t

    def conf_loss(self, pos, neg, conf_data, conf_t):
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)   #[(pos_idx + neg_idx).gt(0)]
        targets_weighted = conf_t[(pos + neg).gt(0)] .long()  #[(pos + neg).gt(0)]

        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False) # .float()

        return loss_c

    def hard_mining(self, conf_data, conf_t, pos, num):
        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)

        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.long().view(-1, 1))


        # Hard Negative Mining
        loss_c[pos.view(-1, 1)] = 0  # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)
        return neg


def loc_loss(pos, loc_data, loc_t):

    pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
    loc_p = loc_data[pos_idx].view(-1, 4)
    loc_t = loc_t[pos_idx].view(-1, 4)
    loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

    return loss_l.float()





















