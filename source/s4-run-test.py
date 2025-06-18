import os
import shutil
from pickle import HIGHEST_PROTOCOL, dump, load

import torch as t
from tqdm import tqdm

from model.network import MIRNet, MPDNet
from preset.config import Config
from util.dataloader import TestImgSet, TestProjSet
from util.dicomio import loadImgMetadata, loadProjMetadata
from util.reconutil import FFTFilter, rebin, rebinFFS, recon


# projection rebinning
def runRebin(study, opt):
    # load metadata
    proj_md_path = os.path.join(opt.proj_dir, study, "proj-params.pkl")
    proj_md = loadProjMetadata(proj_md_path)

    # rebin
    data_path = os.path.join(opt.proj_dir, study)
    save_path = os.path.join(opt.result_dir, study, "rebin-projs")

    if opt.manufacturer == "Siemens":
        rebinFFS(data_path, save_path, proj_md, device=opt.device, desc=study + "-rebin")

    elif opt.manufacturer == "GE":
        rebin(data_path, save_path, proj_md, device=opt.device, desc=study + "-rebin")

    else:
        raise NotImplementedError()


# noise estimation
def runDenoise(study, opt):
    # load model
    mpd_net = MPDNet(n_frames=opt.n_frames_mpd, manufacturer=opt.manufacturer).to(opt.device)
    mpd_net.load()
    mpd_net.eval()

    # make folder
    save_path = os.path.join(opt.result_dir, study, "denoise-projs")
    os.makedirs(save_path, exist_ok=True)

    # dataset for testing
    data_path = os.path.join(opt.result_dir, study, "rebin-projs")
    test_dataset = TestProjSet(data_path)
    test_loader = t.utils.data.DataLoader(test_dataset)

    # make high-pass filter
    with open(os.path.join(data_path, "../", "rebin-params.pkl"), "rb") as file:
        rebin_md = load(file)
    hpf = FFTFilter(rebin_md, kernel=opt.recon_filter, device=opt.device)

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
        # pre-processing
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
            _, pred_noise = mpd_net(ld_proj, ld_noise, t_idx=t_idx)

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
        rebin_pred = hpf.run(rebin_pred)
        rebin_ld = hpf.run(rebin_ld)
        rebin_fd = hpf.run(rebin_fd)

        # get gantry params
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
        os.path.join(opt.result_dir, study, "denoise-projs"),
        os.path.join(opt.result_dir, study, "recon-imgs"),
        proj_md,
        img_md,
        n_interp=opt.n_interp,
        device=opt.device,
        desc=study + "-recon",
    )


# image refinement
def runRefine(study, opt):
    # load model
    mir_net = MIRNet(n_frames=opt.n_frames_mir, manufacturer=opt.manufacturer).to(opt.device)
    mir_net.load()
    mir_net.eval()

    # make folder
    pred_path = os.path.join(opt.result_dir, "prediction", study)
    os.makedirs(pred_path, exist_ok=True)
    gt_path = os.path.join(opt.result_dir, "baseline", study)
    os.makedirs(gt_path, exist_ok=True)

    # dataset for testing
    test_dataset = TestImgSet(os.path.join(opt.result_dir, study, "recon-imgs"))
    test_loader = t.utils.data.DataLoader(test_dataset)

    # temporary buffers
    noise_buffer = []
    ld_buffer = []
    fd_buffer = []
    mask_buffer = []
    for index, (ld_noise, ld_img, fd_img, mask) in tqdm(
        enumerate(test_loader), desc=study + "-refine", total=len(test_loader)
    ):
        # pre-processing
        ld_noise = ld_noise.to(opt.device)
        ld_img = ld_img.to(opt.device)

        noise_buffer.append(ld_noise)
        ld_buffer.append(ld_img)
        fd_buffer.append(fd_img)
        mask_buffer.append(mask)

        if index < (opt.n_frames_mir - 1) or (index + 1 - opt.n_frames_mir) % (opt.n_frames_mir // 2) != 0:
            continue

        noise_buffer = noise_buffer[-opt.n_frames_mir :]
        ld_buffer = ld_buffer[-opt.n_frames_mir :]
        fd_buffer = fd_buffer[-opt.n_frames_mir :]
        mask_buffer = mask_buffer[-opt.n_frames_mir :]

        # convert to tensors
        in_noise = t.stack(noise_buffer, dim=0).unsqueeze(0)
        in_imgs = t.stack(ld_buffer, dim=0).unsqueeze(0)

        # inference
        with t.no_grad():
            mid_img, pred_noise = mir_net(in_imgs, in_noise)

        pred_img = 1000.0 * (mid_img + pred_noise)
        pred_img = t.clip(pred_img, -1024.0, 3071.0).squeeze()

        # get low- and high-dose images
        ld_img = 1000.0 * mid_img
        ld_img = t.clip(ld_img, -1024.0, 3071.0).squeeze()
        fd_img = 1000.0 * fd_buffer[opt.n_frames_mir // 2]
        fd_img = t.clip(fd_img, -1024.0, 3071.0).squeeze()

        gt = {
            "ld": ld_img.cpu().numpy(),
            "fd": fd_img.cpu().numpy(),
            "mask": mask_buffer[opt.n_frames_mir // 2].squeeze().numpy(),
        }

        # save to file
        f_idx = (index + 1 - opt.n_frames_mir) // (opt.n_frames_mir // 2) + 1
        with open(os.path.join(pred_path, "%04d.pkl" % f_idx), "wb") as file:
            dump(pred_img.cpu().numpy(), file, protocol=HIGHEST_PROTOCOL)
        with open(os.path.join(gt_path, "%04d.pkl" % f_idx), "wb") as file:
            dump(gt, file, protocol=HIGHEST_PROTOCOL)


# process each study
def main():
    # load config
    opt = Config(mode="Test")

    # load study list
    study_list = opt.test_list
    study_list.sort()

    for study in study_list:
        # projection rebinning
        runRebin(study, opt)

        # projection denoising
        runDenoise(study, opt)

        # image reconstruction
        runRecon(study, opt)

        # image refinement
        runRefine(study, opt)

        # clean up
        shutil.rmtree(os.path.join(opt.result_dir, study))


if __name__ == "__main__":
    main()
