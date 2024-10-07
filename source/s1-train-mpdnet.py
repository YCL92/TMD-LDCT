import logging
import os
import random

import numpy as np
import torch as t
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from model.network import MPDNet
from preset.config import Config
from util.dataloader import TrainProjSet, ValProjSet


def validate(model, val_loader, opt):
    # set to evaluation mode
    model.eval()

    rmse_list = []
    for index, (ld_noise, ld_projs, fd_projs) in enumerate(val_loader):
        # copy to device
        ld_noise = ld_noise.to(opt.device)
        ld_projs = ld_projs.to(opt.device)
        fd_projs = fd_projs.to(opt.device)

        # run model
        with t.no_grad():
            for t_idx in range(opt.buffer_size):
                in_noise = ld_noise[:, t_idx, :, :, :]
                in_proj = ld_projs[:, t_idx, :, :, :]

                mid_proj, pred_noise = model(in_proj, in_noise, t_idx=t_idx)

        # compute rmse
        pred_proj = mid_proj + pred_noise
        fd_proj = fd_projs[:, opt.buffer_size // 2, :, :, :]
        rmse = t.sqrt(t.nn.functional.mse_loss(pred_proj, fd_proj))
        rmse_list.append(rmse)

    # set to training mode
    model.train(mode=True)
    ave_rmse = t.mean(t.stack(rmse_list)).item()

    return ave_rmse


def main():
    # load config
    opt = Config(mode="MPD-Net")
    print("Manufacturer:", opt.manufacturer)
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
    model = MPDNet(n_frames=opt.n_frames, manufacturer=opt.manufacturer).to(opt.device)

    # dataset for training
    train_dataset = TrainProjSet(opt)
    train_loader = t.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers
    )

    # dataset for validation
    val_dataset = ValProjSet(opt)
    val_loader = t.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=opt.num_workers
    )

    # optimizer
    model_optim = t.optim.Adam(model.parameters(), lr=opt.lr)

    # scheduler
    model_sched = ReduceLROnPlateau(
        model_optim,
        mode="min",
        factor=opt.lr_decay,
        patience=2,
        threshold=1e-5,
        threshold_mode="abs",
        min_lr=opt.lr * opt.lr_decay**2,
        verbose=True,
    )

    # loss function
    l1_loss = t.nn.L1Loss()

    # logging
    logging.basicConfig(
        filename=os.path.join(opt.checkpoint_dir, "history-mpdnet.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info("Starting training...")

    best_rmse = 100.0
    for epoch_idx in range(opt.max_epoch):
        for iter_idx, (ld_noise, ld_projs, fd_projs) in tqdm(
            enumerate(train_loader), desc="Epoch %02d" % (epoch_idx + 1), total=len(train_loader)
        ):
            # copy to device
            ld_noise = ld_noise.to(opt.device)
            ld_projs = ld_projs.to(opt.device)
            fd_projs = fd_projs.to(opt.device)

            # reset gradient
            model_optim.zero_grad()

            # forward pass
            for t_idx in range(opt.buffer_size):
                in_noise = ld_noise[:, t_idx, :, :, :]
                in_proj = ld_projs[:, t_idx, :, :, :]

                mid_proj, pred_noise = model(in_proj, in_noise, t_idx=t_idx)

            # compute loss
            pred_proj = mid_proj + pred_noise
            fd_proj = fd_projs[:, opt.buffer_size // 2, :, :, :]
            pred_loss = l1_loss(pred_proj, fd_proj)

            # update network params
            pred_loss.backward()
            model_optim.step()

            # perform validation and save model if needed
            if (iter_idx + 1) % opt.val_freq == 0:
                cur_rmse = validate(model, val_loader, opt)

                if cur_rmse < best_rmse:
                    model.save()
                    best_rmse = cur_rmse

                logging.info(
                    "Epoch: %d, Iter: %d, Val (current): %.4f, Val (best): %.4f"
                    % (epoch_idx + 1, iter_idx + 1, cur_rmse, best_rmse)
                )

        # update learning rate if needed
        model_sched.step(best_rmse)


if __name__ == "__main__":
    main()
