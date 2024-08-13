import logging
import os
import random

import numpy as np
import torch as t
import torchnet as tnt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from model.network import MIRNet
from preset.config import Config
from util.dataloader import TrainImgSet, ValImgSet
from util.util import MaskL1Loss


def calcPSNR(src_img, tgt_img, mask):
    src_img = src_img.squeeze()
    tgt_img = tgt_img.squeeze()

    assert src_img.size() == tgt_img.size(), "Two images must match in size."

    # normalize to 0-1
    src_data = t.clip(src_img, -1024.0, 3071.0)
    tgt_data = t.clip(tgt_img, -1024.0, 3071.0)
    src_data = (src_data + 1024.0) / 4095.0
    tgt_data = (tgt_data + 1024.0) / 4095.0

    # calculate psnr
    mse = t.sum(mask * (src_data - tgt_data) ** 2) / t.sum(mask)
    psnr = 10 * t.log10(1 / mse)

    return psnr


def validate(model, val_loader, opt):
    # set to evaluation mode
    model.eval()

    psnr_list = []
    for index, (ld_noise, ld_imgs, fd_imgs, masks) in enumerate(val_loader):
        if (index + opt.n_frames // 2) % (opt.n_frames // 2) != 0:
            continue

        # copy to device
        ld_noise = ld_noise.to(opt.device)
        ld_imgs = ld_imgs.to(opt.device)
        fd_imgs = fd_imgs.to(opt.device)
        masks = masks.to(opt.device)

        # ground-truth
        fd_img = fd_imgs[:, opt.n_frames // 2, :, :, :]

        # run model
        with t.no_grad():
            mid_img, pred_noise = model(ld_imgs, ld_noise)

        # calculate PSNR
        pred_img = mid_img + pred_noise
        psnr = calcPSNR(1000.0 * pred_img, 1000.0 * fd_img, masks)
        psnr_list.append(psnr)

    # set to training mode
    model.train(mode=True)
    ave_psnr = t.clip(t.mean(t.stack(psnr_list)), min=0.0).item()

    return ave_psnr


def main():
    # load config
    opt = Config("MIR-Net")
    print("Initial learning rate is %.2e" % opt.lr)

    # setup random environment
    if opt.seed is None:
        seed = t.seed()
    else:
        seed = opt.seed
    t.manual_seed(seed)
    random.seed(seed % 2**32)
    np.random.seed(seed % 2**32)
    print("Use seed", seed)

    # define model
    model = MIRNet(n_frames=opt.n_frames, manufacturer=opt.manufacturer).to(opt.device)

    # dataset for training
    train_dataset = TrainImgSet(opt)
    train_loader = t.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True, pin_memory=True, num_workers=opt.num_workers
    )

    # dataset for validation
    val_dataset = ValImgSet(opt)
    val_loader = t.utils.data.DataLoader(val_dataset, num_workers=opt.num_workers)

    # optimizer
    model_optim = t.optim.Adam(model.parameters(), lr=opt.lr)

    # scheduler
    model_sched = ReduceLROnPlateau(
        model_optim,
        mode="max",
        factor=opt.lr_decay,
        patience=50,
        threshold=0.01,
        threshold_mode="abs",
        min_lr=opt.lr * opt.lr_decay**2,
        verbose=True,
    )

    # loss function
    mse_loss = t.nn.MSELoss()
    l1_loss = MaskL1Loss()

    # logging
    logging.basicConfig(
        filename=os.path.join(opt.checkpoint_dir, "history-mirnet.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info("Starting training...")
    model_meter = tnt.meter.AverageValueMeter()

    best_psnr = 0.0
    for epoch_idx in tqdm(range(opt.max_epoch), total=opt.max_epoch):

        for ld_noise, ld_imgs, fd_imgs, masks in train_loader:
            # copy to device
            ld_noise = ld_noise.to(opt.device)
            ld_imgs = ld_imgs.to(opt.device)
            fd_imgs = fd_imgs.to(opt.device)
            masks = masks.to(opt.device)

            # get ground-truth
            fd_img = fd_imgs[:, opt.n_frames // 2, :, :, :]
            mask = masks[:, opt.n_frames // 2, :, :, :]

            # reset gradient
            model_optim.zero_grad()

            # forward pass
            mid_img, pred_noise = model(ld_imgs, ld_noise)

            # compute loss
            pred_img = mid_img + pred_noise
            pred_loss = l1_loss(1000.0 * pred_img, 1000.0 * fd_img, mask)

            # update network params
            pred_loss.backward()
            model_optim.step()

            # add to loss meter for logging
            model_meter.add(pred_loss.item())

        # perform validation and save model if needed
        cur_psnr = validate(model, val_loader, opt)

        if cur_psnr > best_psnr:
            model.save()
            best_psnr = cur_psnr

        logging.info("Epoch: %d, Val (current): %.2f, Val (best): %.2f" % (epoch_idx + 1, cur_psnr, best_psnr))

        # update learning rate if needed
        model_sched.step(best_psnr)


if __name__ == "__main__":
    main()
