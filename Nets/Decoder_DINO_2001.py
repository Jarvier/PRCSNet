import torch
import torch.nn as nn
import torch.nn.functional as F
from mpmath.libmp import normalize
# from pygments.styles.dracula import foreground
from sqlalchemy.testing import is_none

from SwinTransformer import SwinT,SwinS,SwinB

from modules.Crow import apply_crow_aggregation
from modules.decoder_module import PAA_d, PAA_ed_ori, PAA_attd
from modules.context_module import QKV, PAA_fu
from  modules.layers import SelfAttention
from DINO_model.make_dino_model import get_dinov2_model,get_dinov3_model
from DINO_model.hooks import get_self_attention, process_self_attention, get_second_last_out, get_vit_out, get_dinov1_patches, \
    feats, get_second_out

import numpy as np
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

from Nets.Plain_threshold_cluster import ThresholdBasedClusteringV3
from Nets.FeatureThreshold import FeatureThresholdMask,UniversalBackground
from Nets.GraphPropagation2D import GraphPropagation2D,GraphPropagation2DV2,GraphPropagation2DV2_Sparse,GraphPropagation2D_NativeSparse,GraphPropagation2DV3
from Nets.initial_mask_by_sim import initial_mask_by_sim, object_mask_by_dino04, object_mask_by_dino201,object_mask_by_dino201_DINOv2,initial_mask_by_sim_dino18_02,object_mask_by_dino_head
from Nets.initial_mask_by_sim import initial_mask_by_sim_dino201,initial_mask_by_sim_dino201_1,initial_mask_by_sim_dino18_02
from modules.pamr import PAMR



class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, padding='same',
                 bias=False, bn=True, relu=False):
        super(Conv2d, self).__init__()
        if '__iter__' not in dir(kernel_size):
            kernel_size = (kernel_size, kernel_size)
        if '__iter__' not in dir(stride):
            stride = (stride, stride)
        if '__iter__' not in dir(dilation):
            dilation = (dilation, dilation)

        if padding == 'same':
            width_pad_size = kernel_size[0] + (kernel_size[0] - 1) * (dilation[0] - 1)
            height_pad_size = kernel_size[1] + (kernel_size[1] - 1) * (dilation[1] - 1)
        elif padding == 'valid':
            width_pad_size = 0
            height_pad_size = 0
        else:
            if '__iter__' in dir(padding):
                width_pad_size = padding[0] * 2
                height_pad_size = padding[1] * 2
            else:
                width_pad_size = padding * 2
                height_pad_size = padding * 2

        width_pad_size = width_pad_size // 2 + (width_pad_size % 2 - 1)
        height_pad_size = height_pad_size // 2 + (height_pad_size % 2 - 1)
        pad_size = (width_pad_size, height_pad_size)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad_size, dilation, groups, bias=bias)
        self.reset_parameters()

        if bn is True:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None

        if relu is True:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.conv.weight)

def weight_init(net):
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)




from typing import List, Dict


class Decoder_DINO(nn.Module):
    def __init__(self, backbone='B'):
        super(Decoder_DINO, self).__init__()
        swin_channels=[]
        if backbone == 'B':
            self.backbone = SwinB(pretrained=True)
            swin_channels = [128,128,256,512,1024]
        elif backbone == 'S':
            self.backbone = SwinS(pretrained=True)
            swin_channels = [96,96,192,384,768]
        elif backbone == 'T':
            self.backbone = SwinT(pretrained=True)
            swin_channels = [96,96,192,384,768]

        # self.dino_model = get_dinov2_model()
        self.dino_model = get_dinov3_model()


        # self.top_ch_num = swin_channels[-1]
        self.bottom_ch_num = swin_channels[0]
        self.top_ch_num = 1024



        # self.freeze = args.freeze

        self.depth=128


        self.softmax_1d = nn.Softmax(dim=0)
        self.top_rbf_gamma = nn.Parameter(torch.ones([1])*0.6)
        self.logit_scale = nn.Parameter(torch.ones([1])*4.0)

        self.initial_object_by_dino=object_mask_by_dino201()
        self.initial_mask_by_sim_dino=initial_mask_by_sim_dino201()

        self.dec_51 = PAA_ed_ori(self.top_ch_num+self.bottom_ch_num, out_channel=1, depth=self.depth, last_bn=False)


        # weight_init(self)
        self.ret = lambda x, target: F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        self.upsample = lambda img, size: F.interpolate(img, size=size, mode='bilinear', align_corners=True)

        self.freeze=True

    def cuda(self, idx=None):
        if idx is None:
            idx = torch.cuda.current_device()

        self.to(device="cuda:{}".format(idx))
        return self

    def to(self,device):
        super(Decoder_DINO,self).to(device)
        return self


    def forward(self,rgb):#,dino_qkv, dino_vis_embedding):
        # rgb=torch.cat([rgb,rgb,rgb],dim=2)
        B,ni,c,h_ori,w_ori = rgb.size()
        device = rgb.device
        if self.freeze:
            for param in self.backbone.parameters():
                param.requires_grad=False


        dino_qkv=[] #
        dino_embedding = []
        dino_second_last_embedding=[]
        self.dino_model.blocks[-1].attn.qkv.register_forward_hook(get_self_attention)
        self.dino_model.blocks[-2].register_forward_hook(get_second_last_out)
        for i in range(ni):
            tmp_rgb = rgb[:, i, :, :, :]
            dino_out = self.dino_model(tmp_rgb,is_training=True)
            qkv_i = feats['self_attn']
            dino_qkv.append(qkv_i)
            vis_features_i = dino_out['x_norm_patchtokens']
            dino_embedding.append(vis_features_i)
            second_last_i = feats['second_last_out']
            dino_second_last_embedding.append(second_last_i)


        dino_qkv=torch.stack(dino_qkv).transpose(1,0).contiguous() # B N 16 1374 3072
        dino_embedding=torch.stack(dino_embedding).transpose(1,0).contiguous() # B N 1369 1024


        #
        # feature extraction
        A1=[]
        A2=[]
        A3=[]
        A4=[]
        A5=[]
        for i in range(ni):
            A1_r,A2_r,A3_r,A4_r,A5_r = self.backbone(rgb[:,i,:,:,:])
            A1.append(A1_r)
            A2.append(A2_r)
            A3.append(A3_r)
            A4.append(A4_r)
            A5.append(A5_r)
        #
        dino_objectness_masks = self.initial_object_by_dino(dino_qkv) # B N 1 h w tensor
        objectness_masks=dino_objectness_masks

        sim_mask_dino = self.initial_mask_by_sim_dino(dino_embedding, objectness_masks)
        # sim_mask_dino = self.initial_mask_by_sim_dino(dino_embedding) # for ablation study

        initial_co_salient_mask= sim_mask_dino


        # # #
        # #
        detail_logits=[]
        # A5=torch.stack(A5).transpose(1,0).contiguous()
        # A4=torch.stack(A4).transpose(1,0).contiguous()
        # A3=torch.stack(A3).transpose(1,0).contiguous()
        # A2=torch.stack(A2).transpose(1,0).contiguous()
        A1=torch.stack(A1).transpose(1,0).contiguous()
        base_scale=h_ori//224
        for i in range(ni):
            dino_embedding_i=dino_embedding[:,i,:,:].permute(0,2,1).view(B,1024,14*base_scale,14*base_scale)
            # A5_i=A5[:,i,:,:,:]
            # A4_i=A4[:,i,:,:,:]
            # A3_i=A3[:,i,:,:,:]
            # A2_i=A2[:,i,:,:,:]
            A1_i=A1[:,i,:,:,:]
            initial_co_salient_mask_i = initial_co_salient_mask[:,i,:,:,:]
            # masked_top_feature_i=masked_top_feature[:,i,:,:,:]
            apply_mask=True
            scale=h_ori//224
            if apply_mask:
                mask_dino = F.interpolate(initial_co_salient_mask_i,size=[14*scale,14*scale],mode='bilinear')
                # mask_5 = F.interpolate(initial_co_salient_mask_i,size=[7,7],mode='bilinear')
                # mask_4 = F.interpolate(initial_co_salient_mask_i,size=[14,14],mode='bilinear')
                # mask_3 = F.interpolate(initial_co_salient_mask_i, size=[28,28], mode='bilinear')
                # mask_2 = F.interpolate(initial_co_salient_mask_i, size=[56,56], mode='bilinear')
                mask_1 = F.interpolate(initial_co_salient_mask_i, size=[56*scale,56*scale], mode='bilinear') # because swintransformer feature size
                dino_embedding_i=dino_embedding_i*mask_dino
                # A5_i = A5_i * mask_5
                # A4_i = A4_i * mask_4
                # A3_i = A3_i * mask_3
                # A2_i = A2_i * mask_2
                A1_i = A1_i * mask_1
            feature_1,detail_logits_i=self.dec_51([A1_i,dino_embedding_i],112,112)
            # detail_logits_i=F.interpolate(detail_logits_i,scale_factor=2,mode='bilinear')
            detail_logits.append(detail_logits_i)
        detail_co_salient_logits=torch.stack(detail_logits).transpose(1,0).contiguous()
        # objects_logits=torch.stack(objectness_masks).transpose(1,0)

        initial_co_salient_mask = initial_co_salient_mask.squeeze(2)
        initial_co_salient_mask = F.interpolate(initial_co_salient_mask, size=[112, 112])
        initial_co_salient_mask = initial_co_salient_mask.unsqueeze(2)

        initial_co_salient_mask=torch.clamp(initial_co_salient_mask,0.01,0.99) # for logit transform, avoiding inf
        initial_co_salient_logits=torch.logit(initial_co_salient_mask)

        out={}
        out['logits']=[detail_co_salient_logits,detail_co_salient_logits,initial_co_salient_logits,initial_co_salient_logits,initial_co_salient_logits]

        return out

# Example usage:
# Assuming X is a 3D tensor of activations with dimensions (channels, height, width)
# X = torch.randn(channels, height, width)
# aggregated_feature = apply_crow_aggregation(X)
if __name__ == '__main__':
    rgb=torch.randn([10,3,224,224])
    t=torch.randn([10,1,224,224])
    t_hsv = torch.randn([10,3,224,224])
    rgb=rgb.cuda()
    t=t.cuda()
    t_hsv = t_hsv.cuda()
    rgb2x = F.interpolate(rgb,scale_factor=2)
    t2x = F.interpolate(t,scale_factor=2)
    t2x_hsv = F.interpolate(t_hsv,scale_factor=2)
    rgb2x = rgb2x.cuda()
    t2x = t2x.cuda()
    t2x_hsv = t2x_hsv.cuda()
    model=Decoder_DINO()
    model=model.cuda()
    out=model(rgb,t,rgb2x,t2x)
    print(out)