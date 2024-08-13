import os
import shutil
from pickle import load, dump, HIGHEST_PROTOCOL

import torch as t
from tqdm import tqdm

from model.network import MPDNet
from preset.config import Config
from util.dataloader import TestProjSet
from util.dicomio import loadProjMetadata, loadImgMetadata
from util.reconutil import FFTFilter, recon


# noise estimation
def runDenoise(study, opt):
    # make folder
    save_path = os.path.join(opt.img_dir, study, "denoise-projs")
    os.makedirs(save_path, exist_ok=True)

    # dataset for testing
    data_path = os.path.join(opt.proj_dir, study, "rebin-projs")
    test_dataset = TestProjSet(data_path)
    test_loader = t.utils.data.DataLoader(test_dataset)

    # define model
    model = MPDNet(n_frames=opt.n_frames_mpd, manufacturer=opt.manufacturer).to(opt.device)
    model.load()
    model.eval()

    # make high-pass filter
    with open(os.path.join(data_path, "../", "rebin-params.pkl"), "rb") as file:
        rebin_md = load(file)
    hpFilter = FFTFilter(rebin_md, kernel=opt.recon_filter, device=opt.device)

    # other params
    buffer_size = int(2 * opt.n_frames_mpd - 1)
    mid_idx = int((2 * opt.n_frames_mpd - 1) // 2)

    # temporary buffers
    pred_buffer = []
    ld_buffer = []
    fd_buffer = []
    theta_buffer = []
    zloc_buffer = []
    for t_idx, (ld_noise, ld_proj, fd_proj, rebin_w, theta, zloc) in tqdm(
        enumerate(test_loader), desc=study + "-denoise", total=len(test_loader)
    ):
        # copy to device
        ld_noise = t.swapaxes(ld_noise, 0, 1).to(opt.device)
        ld_proj = t.swapaxes(ld_proj, 0, 1).to(opt.device)
        fd_proj = t.swapaxes(fd_proj, 0, 1).to(opt.device)
        rebin_w = rebin_w.squeeze().to(opt.device)

        ld_buffer.append(ld_proj.squeeze())
        fd_buffer.append(fd_proj.squeeze())
        theta_buffer.append(theta.squeeze())
        zloc_buffer.append(zloc.squeeze())

        # inference
        with t.no_grad():
            _, pred_noise = model(ld_proj, ld_noise, t_idx=t_idx)

        if pred_noise is None:
            continue

        pred_buffer.append(pred_noise.squeeze())

        if len(pred_buffer) < 2:
            continue

        pred_buffer = pred_buffer[-2:]
        ld_buffer = ld_buffer[-(buffer_size + 1) :]
        fd_buffer = fd_buffer[-(buffer_size + 1) :]
        theta_buffer = theta_buffer[-(buffer_size + 1) :]
        zloc_buffer = zloc_buffer[-(buffer_size + 1) :]

        # weighted summation

        rebin_pred = t.sum(rebin_w * t.cat(pred_buffer, dim=0), dim=0)
        rebin_ld = t.sum(rebin_w * t.cat(ld_buffer[mid_idx : mid_idx + 2], dim=0), dim=0)
        rebin_fd = t.sum(rebin_w * t.cat(fd_buffer[mid_idx : mid_idx + 2], dim=0), dim=0)

        # filtering
        rebin_pred = hpFilter.run(rebin_pred)
        rebin_ld = hpFilter.run(rebin_ld)
        rebin_fd = hpFilter.run(rebin_fd)

        # gantry params
        rebin_theta = theta_buffer[mid_idx]
        rebin_zloc = zloc_buffer[mid_idx]

        # save to file
        save_dict = {
            "noise": rebin_pred.cpu(),
            "ld": rebin_ld.cpu(),
            "fd": rebin_fd.cpu(),
            "theta": rebin_theta,
            "zloc": rebin_zloc,
        }

        with open(os.path.join(save_path, "%06d.pkl" % (t_idx - 1 - mid_idx)), "wb") as file:
            dump(save_dict, file, protocol=HIGHEST_PROTOCOL)


# image reconstruction
def runRecon(study, opt):
    # load metadata
    proj_md = loadProjMetadata(os.path.join(opt.proj_dir, study, "proj-params.pkl"))
    img_md = loadImgMetadata(os.path.join(opt.proj_dir, study, "img-params.pkl"))

    # reconstruction
    recon(
        os.path.join(opt.img_dir, study, "denoise-projs"),
        os.path.join(opt.img_dir, study, "recon-imgs"),
        proj_md,
        img_md,
        n_interp=opt.n_interp,
        device=opt.device,
        desc=study + "-recon",
    )


def main():
    # load config
    opt = Config(mode="Test")

    # load study list
    study_list = opt.train_list + opt.val_list
    study_list.sort()

    # process each study
    for study in study_list:
        # step 1: projection denoising
        runDenoise(study, opt)

        # step 2: image reconstruction
        runRecon(study, opt)
        shutil.rmtree(os.path.join(opt.img_dir, study, "denoise-projs"))


if __name__ == "__main__":
    main()
