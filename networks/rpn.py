import torch.nn.functional as F
from torch import nn
import numpy as np
from utils.tools import generate_anchors
from .proposal_layer import ProposalLayer
from config import cfg


class RPN(nn.Module):
    def __init__(self, ratios, scales, c_in=512, c_out=512, feat_stride=16):
        super(RPN, self).__init__()
        self.base_anchors = generate_anchors(cfg.ANCHOR_BASE_SIZE, ratios, scales)
        self.num_of_anchors = len(self.base_anchors)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.cls_layer = nn.Conv2d(c_out, 2 * self.num_of_anchors, 1)
        self.regr_layer = nn.Conv2d(c_out, 4 * self.num_of_anchors, 1)
        self.feat_stride = feat_stride
        self._weight_init()
        self.proposal_layer = ProposalLayer(self)

    def forward(self, feature_map, img_size, scale):
        n, _, h, w = feature_map.size()  # shape(1,512,30,40)
        # print("Feature map in rpn.py",feature_map.size())
        # print(self.base_anchors)
        anchors = self._shift(h, w)  # shape(2400,4)
        # print("Anchor box size :",anchors.shape)
        x = F.relu(self.conv1(feature_map))  # shape(1,512,30,40)

        rpn_regr = self.regr_layer(x)  # shape(1,8,30,40)
        rpn_regr = rpn_regr.permute(0, 2, 3, 1).reshape(n, -1, 4)  # shape(1,2400,4)

        rpn_cls = self.cls_layer(x)  # shape(1,4,30,40)
        rpn_cls = rpn_cls.permute(0, 2, 3, 1)  # shape(1,30,40,4)
        rpn_cls_fg = rpn_cls.reshape(n, h, w, self.num_of_anchors, 2)[:, :, :, :, 1]  # shape(1,30,40,2)
        rpn_cls_fg = rpn_cls_fg.reshape(n, -1)  # shape(1,2400)
        rpn_cls = rpn_cls.reshape(n, -1, 2)  # shape(1,2400,2)
        # Converts RPN outputs into rois
        rois, rois_scores = self.proposal_layer(
            rpn_regr[0].detach().cpu().numpy(),
            rpn_cls_fg[0].detach().cpu().numpy(),
            anchors, img_size, scale
        )

        return rpn_regr, rpn_cls, rois, rois_scores, anchors

    def _shift(self, h, w):
        """
        Generate all anchors in the origin image.

        :param h: int, height of feature map
        :param w: int, width os feature map
        :return: 2d array, shape(h * w * size of base anchors, 4)
        """
        shift_x = np.arange(w) * self.feat_stride
        shift_y = np.arange(h) * self.feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_y.ravel(), shift_x.ravel(),
                            shift_y.ravel(), shift_x.ravel())).transpose()
        all_anchors = self.base_anchors[np.newaxis, :, :] + shifts[:, np.newaxis, :]
        all_anchors = all_anchors.reshape(-1, 4)
        return all_anchors

    def _weight_init(self, mean=0., std=0.01):
        """
        Initialize the weights of conv & cls & regr layer
        """
        for m in [self.conv1, self.cls_layer, self.regr_layer]:
            m.weight.data.normal_(mean, std)
            m.bias.data.zero_()
