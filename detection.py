import torch
from torch.autograd import Function
import torch.nn.functional as F
from utils import decode,nms,pyramidAnchors

class Detect(Function):
    def __init__(self, num_classes, top_k, confidence_thred, nms_thred, device):
        self.num_classes = num_classes
        self.top_k = top_k
        self.confidence_thred = confidence_thred
        self.nms_thred = nms_thred
        self.device = device

    def forward(self, face_conf, face_locdata):
        priors = pyramidAnchors(640)

        face_confdata_0, _ = torch.max(face_conf[:, :, 0:3], dim=2, keepdim=True)
        face_confdata_1 = face_conf[:, :, 3:4]
        face_confdata = F.softmax(torch.cat((face_confdata_0, face_confdata_1), dim=2), dim=2)      # [n, prior_num, 2]
        conf_pred = face_confdata.transpose(2, 1)



        num = face_conf.size(0)
        output = torch.zeros(num, self.top_k, 5)

        prs = torch.Tensor(priors[0]).to(self.device)
        for i in range(1, len(priors)):
            prs = torch.cat((prs, torch.Tensor(priors[i]).to(self.device)), 0)     # [prior_num, 4]


        for i in range(num):
            conf_scores = conf_pred[i].clone()
            c_mask = conf_scores[0].gt(self.confidence_thred)
            scores = conf_scores[0][c_mask]

            scores, ind = torch.sort(scores, dim=0, descending=True)
            print(scores.size(), scores)
            if scores.dim() == 0:
                continue

            decoded_boxes = decode(face_locdata[i], prs)
            l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
            boxes = decoded_boxes[l_mask].view(-1, 4)
            ids, count = nms(boxes, scores, self.nms_thred, self.top_k)

            output[i, :count] = \
                torch.cat((scores[ids[:count]].unsqueeze(1),
                           boxes[ids[:count]]), 1)


        return output







