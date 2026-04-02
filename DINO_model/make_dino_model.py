import os
os.environ['TORCH_HUB_OFFLINE'] = '0'

import argparse
import clip
import json
import math

import requests
# import webdataset as wds
import tarfile
import timm
import torch
import torchvision.transforms as T

from io import BytesIO
from .hooks import get_self_attention, process_self_attention, get_second_last_out, get_vit_out, get_dinov1_patches, \
    feats
# from src.webdatasets_util import cc2coco_format, create_webdataset_tar, read_coco_format_wds
from PIL import Image
from tqdm import tqdm
# from transformers import Blip2Processor, Blip2ForConditionalGeneration, AddedToken


def get_dinov2_model(model_name='dinov2_vitl14_reg',  resize_dim=518, crop_dim=518):
    device = 'cuda' if torch.cuda.is_available else 'cpu'

    # global num_global_tokens, num_patch_tokens, num_tokens, embed_dim, num_attn_heads, scale, batch_size_

    num_global_tokens = 1 if "reg" not in model_name else 5
    num_patch_tokens = crop_dim // 14 * crop_dim // 14
    num_tokens = num_global_tokens + num_patch_tokens
    if 'vitl' in model_name or 'vit_large' in model_name or 'ViT-L' in model_name:
        embed_dim = 1024
    elif 'vitb' in model_name or '_base' in model_name or 'ViT-B' in model_name:
        embed_dim = 768
    elif 'vits' in model_name or 'vit_small' in model_name:
        embed_dim = 384
    else:
        raise Exception("Unknown ViT model")

    scale = 0.125

    # loading the model
    if 'dinov2' in model_name:
        # model_family = 'facebookresearch/dinov2'
        # model = torch.hub.load(model_family, model_name)
        model_family = '/home/fuxin/.cache/torch/hub/facebookresearch_dinov2_main'
        model = torch.hub.load(model_family, model_name,source='local',pretrained=True)

        num_attn_heads = model.num_heads

    elif 'mae' in model_name or 'sam' in model_name or 'clip' in model_name or 'dinov3' in model_name or 'beit':
        model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,  # remove classifier nn.Linear
            img_size=crop_dim
        )
        # the resize dimension will be the one native of the model
        data_config = timm.data.resolve_model_data_config(model)
        image_transforms = timm.data.create_transform(**data_config, is_training=False)

        # adjusting the dimensions
        if 'dinov3' in model_name:
            data_config['input_size'] = (3, crop_dim, crop_dim)
            image_transforms = timm.data.create_transform(**data_config, is_training=False)
            num_patch_tokens = crop_dim // 16 * crop_dim // 16
            num_global_tokens = 5
            num_tokens = num_global_tokens + num_patch_tokens
        elif 'mae' in model_name or 'dino' in model_name or 'beit':
            num_patch_tokens = crop_dim // 16 * crop_dim // 16
            num_tokens = 1 + num_patch_tokens
        elif 'sam' in model_name:
            num_patch_tokens = crop_dim // 16 * crop_dim // 16
            num_tokens = num_patch_tokens
            num_global_tokens = 0
            model.blocks[-1].register_forward_hook(get_vit_out)
        elif 'clip' in model_name:
            crop_dim = resize_dim = 224
            num_patch_tokens = crop_dim // 16 * crop_dim // 16 if 'vit_base' in model_name else crop_dim // 14 * crop_dim // 14
            num_tokens = 1 + num_patch_tokens
        num_attn_heads = model.blocks[-1].attn.num_heads
    elif 'ViT' in model_name:
        # CLIP extraction using clip library
        # use it only for CLS token
        model, image_transforms = clip.load(model_name, device)
        num_attn_heads = model.num_heads
    else:
        raise Exception("Unknown ViT model")

    for param in model.parameters():
        param.requires_grad=False
        # param.requires_grad=True

    return model


def get_dinov3_model(model_name='dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd',  resize_dim=518, crop_dim=518):
    device = 'cuda' if torch.cuda.is_available else 'cpu'

    # global num_global_tokens, num_patch_tokens, num_tokens, embed_dim, num_attn_heads, scale, batch_size_

    num_global_tokens = 1 if "reg" not in model_name else 5
    num_patch_tokens = crop_dim // 16 * crop_dim // 16
    num_tokens = num_global_tokens + num_patch_tokens
    if 'vitl' in model_name or 'vit_large' in model_name or 'ViT-L' in model_name:
        embed_dim = 1024
    elif 'vitb' in model_name or '_base' in model_name or 'ViT-B' in model_name:
        embed_dim = 768
    elif 'vits' in model_name or 'vit_small' in model_name:
        embed_dim = 384
    else:
        raise Exception("Unknown ViT model")

    REPO_DIR = '/home/D/Projects/DINO_v3'
    weight_dir= '/home/D/Projects/DINO_v3_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'

    # DINOv3 ViT models pretrained on web images
    model = torch.hub.load(REPO_DIR, 'dinov3_vitl16', source='local',
                           weights=weight_dir)

    for param in model.parameters():
        param.requires_grad=False
        # param.requires_grad = True

    return model