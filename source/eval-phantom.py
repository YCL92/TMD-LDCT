import os
from pickle import load, dump, HIGHEST_PROTOCOL
from shutil import rmtree

import numpy as np
import torch as t
from torch.nn.functional import grid_sample, relu
from tqdm import tqdm

from config import Config
from model.network import MPDNet, MIRNet
from util.dataloader import ACRProjSet, ACRImgSet
from util.dicomio import loadProjMetadata, loadImgMetadata, save2Dicom
from util.reconutil import FFTFilter


# noise estimation
def runDenoise(dosage, opt):
    # load model
    mpd_net = MPDNet(n_frames=opt.n_frames_mpd, manufacturer=opt.manufacturer).to(opt.device)
    mpd_net.load()
    mpd_net.eval()

    # make folder
    save_dir = os.path.join(opt.result_root, "phantom", dosage + "-denoise-projs")
    os.makedirs(save_dir, exist_ok=True)

    # dataset for testing
    test_dataset = ACRProjSet(opt.acr_root, desc=dosage)
    test_loader = t.utils.data.DataLoader(test_dataset)

    # make high-pass filter
    with open(os.path.join(opt.acr_root, dosage + "-rebin-params.pkl"), "rb") as file:
        rebin_md = load(file)
    hpf = FFTFilter(rebin_md, kernel=opt.recon_filter, device=opt.device)

    # other params
    buffer_size = int(2 * opt.n_frames_mpd - 1)
    mid_idx = int((2 * opt.n_frames_mpd - 1) // 2)

    # temporary buffers
    pred_buffer = []
    in_buffer = []
    theta_buffer = []
    zloc_buffer = []
    for t_idx, (in_noise, in_proj, rebin_w, theta, zloc) in tqdm(
        enumerate(test_loader), desc=dosage + "-denoise", total=len(test_loader)
    ):
        # copy to device
        in_noise = t.swapaxes(in_noise, 0, 1).to(opt.device)
        in_proj = t.swapaxes(in_proj, 0, 1).to(opt.device)

        rebin_w = rebin_w.squeeze().to(opt.device)

        in_buffer.append(in_proj.squeeze())
        theta_buffer.append(theta.squeeze())
        zloc_buffer.append(zloc.squeeze())

        # inference
        with t.no_grad():
            _, pred_noise = mpd_net(in_proj, in_noise, t_idx=t_idx)

        if pred_noise is None:
            continue

        pred_buffer.append(pred_noise.squeeze())

        if len(pred_buffer) < 2:
            continue

        pred_buffer = pred_buffer[-2:]
        in_buffer = in_buffer[-(buffer_size + 1) :]
        theta_buffer = theta_buffer[-(buffer_size + 1) :]
        zloc_buffer = zloc_buffer[-(buffer_size + 1) :]

        # weighted summation

        rebin_pred = t.sum(rebin_w * t.cat(pred_buffer, dim=0), dim=0)
        rebin_ref = t.sum(rebin_w * t.cat(in_buffer[mid_idx : mid_idx + 2], dim=0), dim=0)

        # filtering
        rebin_pred = hpf.run(rebin_pred)
        rebin_ref = hpf.run(rebin_ref)

        # gantry params
        rebin_theta = theta_buffer[mid_idx]
        rebin_zloc = zloc_buffer[mid_idx]

        # save to file
        save_dict = {
            "noise": rebin_pred.cpu(),
            "ref": rebin_ref.cpu(),
            "theta": rebin_theta,
            "zloc": rebin_zloc,
        }

        with open(os.path.join(save_dir, "%06d.pkl" % (t_idx - 1 - mid_idx)), "wb") as file:
            dump(save_dict, file, protocol=HIGHEST_PROTOCOL)


# image reconstruction
def runRecon(dosage, opt):
    # load metadata
    proj_md = loadProjMetadata(os.path.join(opt.acr_root, dosage + "-proj-params.pkl"))
    img_md = loadImgMetadata(os.path.join(opt.acr_root, dosage + "-img-params.pkl"))

    # reconstruction
    data_dir = os.path.join(opt.result_root, "phantom", dosage + "-denoise-projs")
    save_dir = os.path.join(opt.result_root, "phantom", dosage + "-recon-imgs")
    os.makedirs(save_dir, exist_ok=True)

    # apply slice insertion
    recon_list = []
    for i_idx in range(len(img_md["recon_list"]) - 1):
        interval = (img_md["recon_list"][i_idx + 1] - img_md["recon_list"][i_idx]) / opt.n_interp
        for j_idx in range(opt.n_interp):
            recon_list.append(img_md["recon_list"][i_idx] + j_idx * interval)
    recon_list.append(img_md["recon_list"][-1])

    # image and projection attributes
    img_hei = int(img_md["Rows"])
    img_wid = int(img_md["Columns"])
    rfov = float(img_md["ReconstructionDiameter"])
    thickness = float(img_md["SliceThickness"])
    n_rebin_chnls = int(2 * proj_md["n_chnls"])
    n_projs_pi = int(proj_md["n_projs_2pi"] / 4)

    # BP variables
    x_grid, y_grid = t.meshgrid(t.arange(0, img_wid), t.arange(0, img_hei), indexing="xy")
    x_grid = ((rfov / img_wid) * (x_grid - (img_wid - 1) / 2)).to(opt.device)
    y_grid = ((rfov / img_hei) * (y_grid - (img_hei - 1) / 2)).to(opt.device)
    bp_grid = t.zeros((1, img_hei, img_wid, 2), device=opt.device)
    rebin_qtan = proj_md["rebin_qtan"].to(opt.device)
    rebin_wq = proj_md["rebin_wq"].to(opt.device)
    mu_water = proj_md["mu_water"].to(opt.device)

    # expand to 3D without taking extra memory
    x_grid = x_grid.unsqueeze(0)
    y_grid = y_grid.unsqueeze(0)
    rebin_qtan = rebin_qtan.view(-1, 1, 1).expand(-1, img_hei, img_wid)
    rebin_wq = rebin_wq.view(-1, 1, 1).expand(-1, img_hei, img_wid)

    # grab files
    file_list = [item for item in os.listdir(data_dir) if item.endswith(".pkl")]
    file_list.sort()

    # back-projection
    start_idx = 0
    end_idx = len(file_list)
    for s_idx, z_recon in tqdm(enumerate(recon_list), desc=dosage, total=len(recon_list)):
        # temporary buffers
        pred_img_s = t.zeros((img_hei, img_wid, n_projs_pi), device=opt.device)
        ref_img_s = t.zeros((img_hei, img_wid, n_projs_pi), device=opt.device)
        img_w = t.zeros((img_hei, img_wid, n_projs_pi), device=opt.device)

        # loop over all rebined projections
        entry_flag = False
        for f_idx, f_name in enumerate(file_list):
            # make sure file is in the grabing range
            if f_idx < start_idx:
                continue
            elif f_idx > end_idx:
                break

            # load projection, angle, and z location
            file_path = os.path.join(data_dir, f_name)
            with open(file_path, "rb") as file:
                sample = load(file)
            pred_data = sample["noise"].to(opt.device).squeeze().unsqueeze(0)
            ref_data = sample["ref"].to(opt.device).squeeze().unsqueeze(0)
            theta = sample["theta"].to(opt.device).squeeze()
            zloc = -sample["zloc"].to(opt.device).squeeze()  # flipped z-axis

            # calculate distances, weight, and BP results
            p_hat = x_grid * t.sin(theta) - y_grid * t.cos(theta)
            l_hat = t.sqrt(proj_md["r_f"] ** 2 - p_hat**2) - x_grid * t.cos(theta) - y_grid * t.sin(theta)
            dz = zloc - z_recon - proj_md["z_rot"] / (2 * np.pi) * t.arcsin(p_hat / proj_md["r_f"]) + l_hat * rebin_qtan
            h_t = relu(1 - t.abs(dz) / thickness) * rebin_wq

            if t.sum(h_t[:]) > 0:
                if entry_flag is False:
                    entry_flag = True
                    start_idx = max(0, f_idx - n_projs_pi)
                else:
                    end_idx = min(f_idx + n_projs_pi, len(file_list) - 1)

                # 1D interpolation
                bp_grid[0, :, :, 0] = 0
                bp_grid[0, :, :, 1] = p_hat / proj_md["rebin_pmax"]
                pred_bp = grid_sample(
                    pred_data.unsqueeze(-1), bp_grid, mode="bilinear", padding_mode="border", align_corners=True
                ).squeeze()
                ref_bp = grid_sample(
                    ref_data.unsqueeze(-1), bp_grid, mode="bilinear", padding_mode="border", align_corners=True
                ).squeeze()

                # update weight and BP results
                pred_st = h_t * pred_bp
                ref_st = h_t * ref_bp

                p_idx = int(f_idx % n_projs_pi)
                pred_img_s[:, :, p_idx] += t.sum(pred_st, dim=0)
                ref_img_s[:, :, p_idx] += t.sum(ref_st, dim=0)
                img_w[:, :, p_idx] += t.sum(h_t, dim=0)

        # weighted average
        pred_img = np.pi * t.mean(pred_img_s / t.clip(img_w, min=1e-8), dim=-1)
        ref_img = np.pi * t.mean(ref_img_s / t.clip(img_w, min=1e-8), dim=-1)

        # convert to attenuation ratio
        pred_img = pred_img / mu_water
        ref_img = (ref_img - mu_water) / mu_water

        # write to file
        img_path = os.path.join(save_dir, "%04d.pkl" % s_idx)
        out_data = {"noise": pred_img.cpu().numpy(), "img": ref_img.cpu().numpy()}
        with open(img_path, "wb") as file:
            dump(out_data, file, protocol=HIGHEST_PROTOCOL)

    # delete projection data
    rmtree(data_dir)


# image refinement
def runRefine(dosage, opt):
    # load model
    mir_net = MIRNet(n_frames=opt.n_frames_mir, manufacturer=opt.manufacturer).to(opt.device)
    # mir_net.load()
    mir_net.eval()

    # load image metadata
    img_md = loadImgMetadata(os.path.join(opt.acr_root, dosage + "-img-params.pkl"))

    # make folder
    pred_save_dir = os.path.join(opt.result_root, "phantom", dosage + "-pred-imgs")
    os.makedirs(pred_save_dir, exist_ok=True)
    ref_save_dir = os.path.join(opt.result_root, "phantom", dosage + "-ref-imgs")
    os.makedirs(ref_save_dir, exist_ok=True)

    # dataset for testing
    data_dir = os.path.join(opt.result_root, "phantom", dosage + "-recon-imgs")
    test_dataset = ACRImgSet(data_dir)
    test_loader = t.utils.data.DataLoader(test_dataset)

    # temporary buffers
    noise_buffer = []
    in_buffer = []
    for index, (in_noise, ref_img) in tqdm(enumerate(test_loader), desc=dosage + "-refine", total=len(test_loader)):
        # copy to device
        in_noise = in_noise.to(opt.device)
        ref_img = ref_img.to(opt.device)

        noise_buffer.append(in_noise)
        in_buffer.append(ref_img)

        if index < (opt.n_frames_mir - 1) or (index + 1 - opt.n_frames_mir) % (opt.n_frames_mir // 2) != 0:
            continue

        noise_buffer = noise_buffer[-opt.n_frames_mir :]
        in_buffer = in_buffer[-opt.n_frames_mir :]

        # convert to tensors
        in_noise = t.stack(noise_buffer, dim=0).unsqueeze(0)
        in_imgs = t.stack(in_buffer, dim=0).unsqueeze(0)

        # inference
        with t.no_grad():
            mid_img, pred_noise = mir_net(in_imgs, in_noise)

        # prediction
        pred_img = 1000.0 * (mid_img + pred_noise)
        pred_img = t.clip(pred_img, -1024, 3071).squeeze()

        # get low- and high-dose images
        ref_img = 1000.0 * mid_img
        ref_img = t.clip(ref_img, -1024, 3071).squeeze()

        # save to file
        f_idx = index + 1
        save_path = os.path.join(pred_save_dir, "%04d.dcm" % f_idx)
        save2Dicom(pred_img.cpu().numpy(), f_idx, img_md, save_path, desc=dosage)
        save_path = os.path.join(ref_save_dir, "%04d.dcm" % f_idx)
        save2Dicom(ref_img.cpu().numpy(), f_idx, img_md, save_path, desc=dosage)

    # delete reconstruction data
    rmtree(data_dir)


# process each dosage
def main():
    # load config
    opt = Config(mode="Test")

    for dosage in ["full-dose", "low-dose"]:
        # step 2: projection denoising
        runDenoise(dosage, opt)

        # step 3: image reconstruction
        runRecon(dosage, opt)

        # step 4: image refinement
        runRefine(dosage, opt)


if __name__ == "__main__":
    main()
