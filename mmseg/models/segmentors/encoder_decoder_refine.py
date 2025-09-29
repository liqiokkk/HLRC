import torch
from torch import nn
from torch.nn import functional as F
from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder
from mmcv.runner import auto_fp16
import numpy as np

class HaarDownsampling(nn.Module):
    def __init__(self, channel_in):
        super(HaarDownsampling, self).__init__()
        self.channel_in = channel_in

        self.haar_weights = torch.ones(4, 1, 2, 2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.channel_in, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

    def forward(self, x, rev=False):
        if not rev:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(1/16.)

            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.channel_in) / 4.0
            out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
            return out
        else:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(16.)

            out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups = self.channel_in)


@SEGMENTORS.register_module()
class EncoderDecoderRefine(EncoderDecoder):
    def __init__(self,
                 backbone,
                 decode_head,
                 down_ratio=0.25,
                 refine_input_ratio=1.,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        self.down_ratio = down_ratio
        self.refine_input_ratio = refine_input_ratio
        super(EncoderDecoderRefine, self).__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

    def _init_decode_head(self, decode_head):
        assert isinstance(decode_head, list)
        self.decode_head = builder.build_head(decode_head[0])  
        self.refine_head = builder.build_head(decode_head[1])  
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.haar = HaarDownsampling(3)

    def encode_decode(self, img, img_metas):
        img_os2 = self.haar(img)
        img_os2 = self.haar(img_os2[:,:3,:,:])[:,:3,:,:]
        if self.refine_input_ratio == 1.:
            img_refine = img
        elif self.refine_input_ratio < 1.:
            img_refine = nn.functional.interpolate(img, size=[int(img.shape[-2] * self.refine_input_ratio), int(img.shape[-1] * self.refine_input_ratio)])

        x = self.extract_feat(img_os2)
        out_g, prev_outputs = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        out = self.refine_head.forward_test(img_refine, prev_outputs, img_metas, self.test_cfg)
        
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out
    
    def forward_train(self, img, img_metas, gt_semantic_seg):
        img_os2 = self.haar(img)
        img_os2 = self.haar(img_os2[:,:3,:,:])[:,:3,:,:]
        if self.refine_input_ratio == 1.:
            img_refine = img
        elif self.refine_input_ratio < 1.:
            img_refine = nn.functional.interpolate(img, size=[int(img.shape[-2]*self.refine_input_ratio), int(img.shape[-1]*self.refine_input_ratio)])
        
        x = self.extract_feat(img_os2)
        
        losses = dict()
        loss_decode = self._decode_head_forward_train(x, img_refine, img_metas, gt_semantic_seg)
        losses.update(loss_decode) 
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)

        return losses

    def _decode_head_forward_train(self, x, img, img_metas, gt_semantic_seg):
        losses = dict()
        loss_decode, prev_features = self.decode_head.forward_train(x, img_metas, gt_semantic_seg, self.train_cfg)
        losses.update(add_prefix(loss_decode, 'decode'))
        loss_refine, cv_loss, *loss_region_list = self.refine_head.forward_train(img, prev_features, img_metas, gt_semantic_seg, self.train_cfg)
        losses.update(add_prefix(loss_refine, 'refine'))
        losses.update(add_prefix(cv_loss, 'cv'))

        if loss_region_list[0] is not None:
            j = 1
            for loss_aux in loss_region_list:
                losses.update(add_prefix(loss_aux, 'aux_' + str(j)))
                j += 1
        return losses
