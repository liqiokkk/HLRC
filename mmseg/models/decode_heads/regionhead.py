import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule, trunc_normal_init
from mmcv.utils import to_2tuple

from ..backbones.efficientvit import EfficientViTBackbone
from ..builder import HEADS
from .cascade_decode_head import BaseCascadeDecodeHead
from .decode_head import BaseDecodeHead
from .stdc import ShallowNet
import einops
from ..utils.FELC import FELC

class SegmentationHead(nn.Module):
    def __init__(self, conv_cfg, norm_cfg, act_cfg, in_channels, mid_channels, n_classes, *args, **kwargs):
        super(SegmentationHead, self).__init__()

        self.conv_bn_relu = ConvModule(in_channels, mid_channels, 3,
                                       stride=1,
                                       padding=1,
                                       conv_cfg=conv_cfg,
                                       norm_cfg=norm_cfg,
                                       act_cfg=act_cfg)

        self.conv_out = nn.Conv2d(mid_channels, n_classes, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.conv_bn_relu(x)
        x = self.conv_out(x)
        return x

class Lap_Pyramid_Conv(nn.Module):
    def __init__(self, num_high=3, gauss_chl=3):
        super(Lap_Pyramid_Conv, self).__init__()

        self.num_high = num_high
        self.gauss_chl = gauss_chl
        self.kernel = self.gauss_kernel()

    def gauss_kernel(self, device=torch.device('cuda')):
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]])
        kernel /= 256.
        kernel = kernel.repeat(self.gauss_chl, 1, 1, 1)
        kernel = kernel.to(device)
        return kernel

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def upsample(self, x):
        cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
        cc = cc.permute(0, 1, 3, 2)
        cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
        x_up = cc.permute(0, 1, 3, 2)
        return self.conv_gauss(x_up, 4 * self.kernel)

    def conv_gauss(self, img, kernel):
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
        return out

    def pyramid_decom(self, img):
        current = img
        pyr = []
        for _ in range(self.num_high):
            filtered = self.conv_gauss(current, self.kernel)
            down = self.downsample(filtered)
            up = self.upsample(down)
            if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
                up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
            diff = current - up
            pyr.append(diff)
            current = down
        return pyr

class SRDecoder(nn.Module):
    # super resolution decoder
    def __init__(self, conv_cfg, norm_cfg, act_cfg, channels=128, up_lists=[2, 2, 2]):
        super(SRDecoder, self).__init__()
        self.conv1 = ConvModule(channels, channels // 2, 3, stride=1, padding=1, conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.up1 = nn.Upsample(scale_factor=up_lists[0])
        self.conv2 = ConvModule(channels // 2, channels // 2, 3, stride=1, padding=1, conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.up2 = nn.Upsample(scale_factor=up_lists[1])
        self.conv3 = ConvModule(channels // 2, channels, 3, stride=1, padding=1, conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.up3 = nn.Upsample(scale_factor=up_lists[2])
        self.conv_sr = SegmentationHead(conv_cfg, norm_cfg, act_cfg, channels, channels // 2, 3, kernel_size=1)

    def forward(self, x, fa=False):
        x = self.up1(x)
        x = self.conv1(x)
        x = self.up2(x)
        x = self.conv2(x)
        x = self.up3(x)
        feats = self.conv3(x)
        outs = self.conv_sr(feats)
        if fa:
            return feats, outs
        else:
            return outs

class Reducer(nn.Module):
    # Reduce channel (typically to 128)
    def __init__(self, in_channels=512, reduce=128, bn_relu=True):
        super(Reducer, self).__init__()
        self.bn_relu = bn_relu
        self.conv1 = nn.Conv2d(in_channels, reduce, 1, bias=False)
        if self.bn_relu:
            self.bn1 = nn.BatchNorm2d(reduce)

    def forward(self, x):

        x = self.conv1(x)
        if self.bn_relu:
            x = self.bn1(x)
            x = F.relu(x)

        return x

class Top2MLP(nn.Module):
    def __init__(self, hidden_dim, ffn_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.w = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim, bias=False), nn.ReLU(),
                              nn.Linear(self.hidden_dim, self.hidden_dim, bias=False), nn.ReLU(),
                              nn.Linear(self.hidden_dim, self.ffn_dim, bias=False))

    def forward(self, hidden_states):
        current_hidden_states = self.w(hidden_states)
        return current_hidden_states


class MoeBlock(nn.Module):
    def __init__(self, hidden_dim, ffn_dim, num_local_experts, num_experts_per_tok):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_experts = num_local_experts
        self.top_k = num_experts_per_tok
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        self.experts = nn.ModuleList([Top2MLP(hidden_dim, ffn_dim) for _ in range(self.num_experts)])

    def cv_squared(self, x):
        x = x.sum(0)
        eps = 1e-10
        return x.float().var() / (x.float().mean()**2 + eps)    
        
    def forward(self, hidden_states, train_flag=False):
        b, c, h, w = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 3, 1)
        hidden_states = hidden_states.view(b, -1, c)
        
        batch_size, sequence_length, hidden_dim = hidden_states.shape   
        hidden_states = hidden_states.reshape(-1, hidden_dim)              
        router_logits = self.gate(hidden_states)                       

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)             
        if train_flag:
            cv_loss = dict()
            cv_loss['cv_loss'] = self.cv_squared(routing_weights) * 0.01
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1) 
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)                       
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, self.ffn_dim), dtype=hidden_states.dtype, device=hidden_states.device     
        )

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)  

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx]) 
            if top_x.shape[0] == 0:
                continue
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)                          
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]  

            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, self.ffn_dim)
        final_hidden_states = final_hidden_states.reshape(b, h, w, self.ffn_dim)
        final_hidden_states = final_hidden_states.permute(0, 3, 1, 2)
        if train_flag:
            return final_hidden_states, cv_loss
        else:
            return final_hidden_states
    
class Region(BaseDecodeHead):
    def __init__(self, in_channels, channels, num_classes, expert_number=-1, tok_k=-1, region_res=(4, 4), norm_cfg=None, act_cfg=dict(type='ReLU'), init_cfg={}, *args, **kwargs):
        super(Region, self).__init__(in_channels, channels, num_classes=num_classes, norm_cfg=norm_cfg,
            act_cfg=act_cfg, init_cfg=init_cfg, *args, **kwargs)

        self.region_res = to_2tuple(region_res)

        self.affinity_head = nn.Sequential(
            DepthwiseSeparableConvModule(
                in_channels, channels, kernel_size=3, padding=1, act_cfg=act_cfg, norm_cfg=norm_cfg),
            ConvModule(
                channels, channels, kernel_size=1, act_cfg=None)
        )

        self.rpae = MoeBlock(channels, 9*self.region_res[0]*self.region_res[1], expert_number, tok_k)
        self.act = nn.ReLU()

    def init_weights(self):
        super(Region, self).init_weights()
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=.0)
        assert all(self.affinity_head[-1].conv.bias == 0)

    def forward_affinity(self, x, train_flag=False):
        self._device = x.device
        B, _, H, W = x.shape

        # get affinity
        x = x.contiguous()
        affinity = self.affinity_head(x)
        ### RPAE
        if train_flag:
            affinity, cv_loss = self.rpae(affinity, train_flag)
        else:
            affinity = self.rpae(affinity)
        ### 4*4 is 1 patch, 9 is neighborhood for 3*3 token
        affinity = affinity.reshape(B, 9, *self.region_res, H, W)  

        # handle borders
        affinity[:, :3, :, :, 0, :] = 0
        affinity[:, -3:, :, :, -1, :] = 0
        affinity[:, ::3, :, :, :, 0] = 0
        affinity[:, 2::3, :, :, :, -1] = 0

        affinity = self.act(affinity)
        if train_flag:
            return affinity, cv_loss
        else:
            return affinity

    def forward(self, x_mid, token_logits, train_flag=False):
        B, _, H, W = token_logits.shape
        ### RPB
        if train_flag:
            affinity, cv_loss = self.forward_affinity(x_mid, train_flag) 
        else:
            affinity = self.forward_affinity(x_mid)
        ### SAB
        token_logits = F.unfold(token_logits, kernel_size=3, padding=1).reshape(B, -1, 9, H, W) 
        token_logits = einops.rearrange(token_logits, 'B C n H W -> B H W n C')  

        affinity = einops.rearrange(affinity, 'B n h w H W -> B H W (h w) n')  
        seg_logits = (affinity @ token_logits).reshape(B, H, W, *self.region_res, -1)  
        seg_logits = einops.rearrange(seg_logits, 'B H W h w C -> B C (H h) (W w)')  
        
        if train_flag:
            return seg_logits, cv_loss
        else:
            return seg_logits

@HEADS.register_module()
class RegionHead(BaseCascadeDecodeHead):
    def __init__(self, region_channels, sr_channels=128, effvit='pretrain/b2-r224.pt', expert_number=-1, tok_k=-1, **kwargs):
        super(RegionHead, self).__init__(**kwargs)
        self.conv_seg = SegmentationHead(self.conv_cfg, self.norm_cfg, self.act_cfg, self.channels+96,
                                         self.channels // 16, self.num_classes, kernel_size=1)
        # region
        self.region_channels = region_channels
        
        self.efficientvit = EfficientViTBackbone(inchannels=9, pretrained=effvit)
        
        self.lap_prymaid_conv = Lap_Pyramid_Conv(num_high=2)
        self.region = Region(self.in_channels, self.region_channels, self.num_classes, expert_number, tok_k, norm_cfg=self.norm_cfg) 
        self.sr_channels = sr_channels
        if self.sr_channels is not None:
            self.reduce = Reducer(self.channels, sr_channels)
            self.sr_decoder = SRDecoder(self.conv_cfg, self.norm_cfg, self.act_cfg, channels=sr_channels, up_lists=[4, 2, 2])
            
        self.spa_branch = EfficientViTBackbone(inchannels=3, pretrained='pretrain/b2-r224.pt')

        self.compression = nn.Conv2d(192+96, 96, kernel_size=1)
        self.felc_8 = FELC(96)
        self.felc_16 = FELC(192)
        
    def forward(self, inputs, prev_output, train_flag=True):
        prymaid_results = self.lap_prymaid_conv.pyramid_decom(inputs)
        high_residual_1 = prymaid_results[0]
        high_residual_2 = F.interpolate(prymaid_results[1], prymaid_results[0].size()[2:], mode='bilinear',
                                        align_corners=False)
        high_residual_input = torch.cat([inputs, high_residual_1, high_residual_2], dim=1)
        ### RPB
        shallow_feat16 = self.efficientvit(high_residual_input)['stage_final']

        if train_flag:
            output, cv_loss = self.region(shallow_feat16, prev_output, train_flag)
        else:
            output = self.region(shallow_feat16, prev_output)   
        
        ### LPB
        spa_feats = self.spa_branch(inputs)
        spa_feat_16 = spa_feats['stage_final']
        spa_feat_8 = spa_feats['stage2']
        spa_feat_16 = self.felc_16(spa_feat_16)
        spa_feat_8 = self.felc_8(spa_feat_8)
        spa_feat_16 = F.interpolate(spa_feat_16, size=spa_feat_8.size()[2:], mode='bilinear', align_corners=False)
        spa_feat = self.compression(torch.cat([spa_feat_8, spa_feat_16], dim=1))
        spa_feat = F.interpolate(spa_feat, size=output.size()[2:], mode='bilinear', align_corners=False)

        output = torch.cat([output, spa_feat], dim=1)  
        output = self.conv_seg(output)

        if train_flag:
            if self.sr_channels is not None:
                deep_feat = self.reduce(prev_output)
                feats, output_sr = self.sr_decoder(deep_feat, True)
                losses_re = self.image_recon_loss(high_residual_1 + high_residual_2, output_sr, re_weight=0.1) 
                losses_fa = self.feature_affinity_loss(deep_feat, feats)
                return output, cv_loss, losses_re, losses_fa
            else:
                return output, cv_loss
        else:
            return output
    
    def image_recon_loss(self, img, pred, re_weight=0.5):
        loss = dict()
        if pred.size()[2:] != img.size()[2:]:
            pred = F.interpolate(pred, img.size()[2:], mode='bilinear', align_corners=False)
        recon_loss = F.mse_loss(pred, img) * re_weight
        loss['recon_losses'] = recon_loss
        return loss

    def feature_affinity_loss(self, seg_feats, sr_feats, fa_weight=1., eps=1e-6):
        if seg_feats.size()[2:] != sr_feats.size()[2:]:
            sr_feats = F.interpolate(sr_feats, seg_feats.size()[2:], mode='bilinear', align_corners=False)
        loss = dict()
        # flatten:
        seg_feats_flatten = torch.flatten(seg_feats, start_dim=2)
        sr_feats_flatten = torch.flatten(sr_feats, start_dim=2)
        # L2 norm 
        seg_norm = torch.norm(seg_feats_flatten, p=2, dim=2, keepdim=True)
        sr_norm = torch.norm(sr_feats_flatten, p=2, dim=2, keepdim=True)
        # similiarity
        seg_feats_flatten = seg_feats_flatten / (seg_norm + eps)
        sr_feats_flatten = sr_feats_flatten / (sr_norm + eps)
        seg_sim = torch.matmul(seg_feats_flatten.permute(0, 2, 1), seg_feats_flatten)
        sr_sim = torch.matmul(sr_feats_flatten.permute(0, 2, 1), sr_feats_flatten)
        # L1 loss
        loss['fa_loss'] = F.l1_loss(seg_sim, sr_sim.detach()) * fa_weight
        return loss

    def forward_train(self, inputs, prev_output, img_metas, gt_semantic_seg, train_cfg):
        if self.sr_channels is not None:
            seg_logits, cv_loss, losses_recon, losses_fa = self.forward(inputs, prev_output)
            losses = self.losses(seg_logits, gt_semantic_seg)
            return losses, cv_loss, losses_recon, losses_fa
        else:
            seg_logits, cv_loss = self.forward(inputs, prev_output)
            losses = self.losses(seg_logits, gt_semantic_seg)
            return losses, cv_loss, None

    def forward_test(self, inputs, prev_output, img_metas, test_cfg):
        return self.forward(inputs, prev_output, False)
