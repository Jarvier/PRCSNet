import torch
import torch.nn.functional as F
import torch.nn as nn
import cv2
import numpy as np

def tensor_bound(img, k_size):
    B, C, H, W = img.shape
    pad = int((k_size - 1) // 2)
    img_pad = F.pad(img, pad=[pad, pad, pad, pad], mode='constant', value=0)
    # unfold in the second and third dimensions
    patches = img_pad.unfold(2, k_size, 1).unfold(3, k_size, 1)
    corrosion, _ = torch.min(patches.contiguous().view(B, C, H, W, -1), dim=-1)
    inflation, _ = torch.max(patches.contiguous().view(B, C, H, W, -1), dim=-1)
    return inflation - corrosion

class edge_loss(nn.Module):
    def __init__(self,ksize=3):
        super(edge_loss,self).__init__()
        self.ksize = ksize
        self.BCE_loss = nn.BCEWithLogitsLoss()
    def forward(self,x,g):
        b,c,h,w=g.size()
        x = torch.sigmoid(x)
        x = F.interpolate(x,size=[h,w],mode='bilinear')
        x_edge=tensor_bound(x,self.ksize)
        g_edge = tensor_bound(g,self.ksize)
        loss=self.BCE_loss(x_edge,g_edge)
        return loss

class background_feature_loss(nn.Module):
    def __init__(self,):
        super(background_feature_loss,self).__init__()
    def to(self, device):
        super(background_feature_loss, self).to(device)
        return self
    def forward(self,A5,background_tensor):
        loss = 0
        for i in range(25):
            top_feature_i = A5[i]
            normalized_top_feature_i = F.normalize(top_feature_i,dim=1)
            B,top_c,top_h,top_w=top_feature_i.size()
            normalized_top_feature_i_corners = normalized_top_feature_i[:, :, [0, 0, top_h-1,top_w-1], [0, top_h-1, 0, top_w-1]].view(B, top_c, 2, 2)
            tmp_weight = background_tensor.unsqueeze(-1).unsqueeze(-1)
            tmp_weight = F.normalize(tmp_weight, dim=1)
            correlation_coefficient_i = F.conv2d(normalized_top_feature_i_corners,tmp_weight)
            tmp_value = 1.0-correlation_coefficient_i
            loss+=tmp_value.sum()
        loss = loss/ (25.0*4.0)
        return loss

def bce_loss(pred, mask, reduction='none'):
    bce = F.binary_cross_entropy(pred, mask, reduction=reduction)
    return bce

def bce_loss_with_logits(pred, mask, reduction='none'):
    bce = F.binary_cross_entropy_with_logits(pred,mask,reduction=reduction)
    return bce
def iou_loss(pred, mask, reduction='none'):
    inter = pred * mask
    union = pred + mask
    iou = 1 - (inter + 1) / (union - inter + 1)

    if reduction == 'mean':
        iou = iou.mean()
    # print("iou:",iou)
    return iou
def weighted_bce_loss(pred, mask, reduction='none'):
    weight = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    weight = weight.flatten()

    bce = weight * bce_loss(pred, mask, reduction='none').flatten()

    if reduction == 'mean':
        bce = bce.mean()
    print("bce:", bce)
    return bce

def weighted_bce_loss_with_logits(pred, mask, reduction='none'):
    weight = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    weight = weight.flatten()

    bce = weight * bce_loss_with_logits(pred, mask, reduction='none').flatten()

    if reduction == 'mean':
        bce = bce.mean()
    # print("bce:", bce)
    return bce


def iou_loss_with_logits(pred, mask, reduction='none'):
    return iou_loss(torch.sigmoid(pred), mask, reduction=reduction)


class my_pool(nn.Module):
    def __init__(self, down_factor=0):
        super(my_pool, self).__init__()
        self.layers = []
        self.down_factor = down_factor
        for _ in range(self.down_factor):
            self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

    def forward(self, x):
        ppool = nn.Sequential(*self.layers)
        return ppool(x)

class ImagePyramid:
    def __init__(self, ksize=7, sigma=1, channels=1):
        self.ksize = ksize
        self.sigma = sigma
        self.channels = channels

        k = cv2.getGaussianKernel(ksize, sigma)
        k = np.outer(k, k)
        k = torch.tensor(k).float()
        self.kernel = k.repeat(channels, 1, 1, 1)

    def to(self, device):
        self.kernel = self.kernel.to(device)
        return self

    def cuda(self, idx=None):
        if idx is None:
            idx = torch.cuda.current_device()

        self.to(device="cuda:{}".format(idx))
        return self

    def expand(self, x):
        z = torch.zeros_like(x)
        x = torch.cat([x, z, z, z], dim=1)
        x = F.pixel_shuffle(x, 2)
        x = F.pad(x, (self.ksize // 2,) * 4, mode='reflect')
        x = F.conv2d(x, self.kernel * 4, groups=self.channels)
        return x

    def reduce(self, x):
        x = F.pad(x, (self.ksize // 2,) * 4, mode='reflect')
        x = F.conv2d(x, self.kernel, groups=self.channels)
        x = x[:, :, ::2, ::2]
        return x


class LG_loss(nn.Module):
    def __init__(self):
        super(LG_loss, self).__init__()
        self.image_pyramid = ImagePyramid(ksize=3)
        self.sod_loss_fn = lambda x, y: weighted_bce_loss_with_logits(x, y, reduction='mean') + iou_loss_with_logits(x,
                                                                                                                     y,
                                                                                                                     reduction='mean')

        self.edge_loss = edge_loss(ksize=3)
        self.pc_loss_fn = nn.L1Loss()
        self.des = lambda x, size: F.interpolate(x, size=size, mode='nearest')
        self.down_max_pool = my_pool(down_factor=4)
    def to(self, device):
        self.image_pyramid.to(device)
        super(LG_loss, self).to(device)
        return self

    def forward(self, predictions, gt):
        loss=0
        b,ni,c,H,W = gt.size()
        d1 = predictions[0] # 112
        d2 = predictions[1] # 56
        d3 = predictions[2] # 28
        d4 = predictions[3] # 14
        d5 = predictions[4] # 7
        # y1=gt
        gt=gt.view(-1,c,H,W)
        d1=d1.reshape(-1,c,H//2,W//2) # let imgs to batch
        d2=d2.reshape(-1,c,H//4,W//4) # let imgs to batch
        # d3=d3.reshape(-1,c,H//8,W//8)
        d4=d4.reshape(-1,c,H//16,W//16)
        y1 = self.image_pyramid.reduce(gt)
        y2 = self.image_pyramid.reduce(y1)
        y3 = self.image_pyramid.reduce(y2)
        y4 = self.image_pyramid.reduce(y3)
        # y5 = self.image_pyramid.reduce(y4)

        y4 = self.down_max_pool(gt)

        # loss = self.pc_loss_fn(self.des(d3, (H, W)), self.des(self.image_pyramid.reduce(d2), (H, W)).detach()) * 0.0001
        # loss += self.pc_loss_fn(self.des(d2, (H, W)), self.des(self.image_pyramid.reduce(d1), (H, W)).detach()) * 0.0001
        # loss += self.pc_loss_fn(self.des(d1, (H, W)), self.des(self.image_pyramid.reduce(d0), (H, W)).detach()) * 0.0001

        # loss = self.sod_loss_fn(self.des(d3, (H, W)), self.des(y3, (H, W)))
        # loss += self.sod_loss_fn(self.des(d2, (H, W)), self.des(y2, (H, W)))
        # loss += self.sod_loss_fn(self.des(d1, (H, W)), self.des(y1, (H, W)))
        # loss += self.sod_loss_fn(self.des(d0, (H, W)), self.des(y, (H, W)))
        #
        # loss += self.sod_loss_fn(d5, y5)
        # loss += self.sod_loss_fn(d4, y4)
        # loss +=self.sod_loss_fn(d3,y3)
        loss += self.sod_loss_fn(d2, y2)
        # loss += self.sod_loss_fn(d1, y1)
        # loss += self.edge_loss(d1,y1) * 0.1

        return loss