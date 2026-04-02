# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import dis_util.misc as misc
import dis_util.lr_sched as lr_sched

from loss_dino import LG_loss,background_feature_loss



def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    # for data_iter_step, (samples, gts, tis, ti_hsv,image2x, ti2x, ti2x_hsv) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
    for data_iter_step, (rgb, gts,cls) in enumerate(
                metric_logger.log_every(data_loader, print_freq, header)):  #, dino_qkv, dino_vis_embedding

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # samples = torch.cat([samples,gts],dim=1)
        # top = top.to(device, non_blocking=True) #
        # top = top.to(device) #
        # bottom = bottom.to(device) #
        gts = gts.to(device)
        rgb = rgb.to(device)
        # dino_qkv = dino_qkv.to(device)
        # dino_vis_embedding = dino_vis_embedding.to(device)
        loss = LG_loss().to(device)
        # my_edge_loss = edge_loss(ksize=3).to(device)
        # my_bg_loss = background_feature_loss().to(device)
        # print(cls)

        with torch.amp.autocast('cuda'): #s
            out = model(rgb)#,dino_qkv,dino_vis_embedding) #
            loss1 = loss(out['logits'], gts)
            # loss_edge = my_edge_loss(out1, gts)
            loss = loss1


        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
            # print("Loss is NaN/inf, skipping this batch")
            # continue  # 跳过当前batch，进入下一次迭代

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=1, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}