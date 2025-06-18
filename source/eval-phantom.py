import os
from pickle import HIGHEST_PROTOCOL, dump, load
from shutil import rmtree

import numpy as np
import torch as t
from torch.nn.functional import grid_sample, relu
from tqdm import tqdm

from model.network import MIRNet, MPDNet
from preset.config import Config
from util.dataloader import ACRImgSet, ACRProjSet
from util.dicomio import loadImgMetadata, loadProjMetadata, save2Dicom
from util.reconutil import FFTFilter
from util.util import calcMSE, calcSSIM


# full-dose projection rebinning
def runRebinFD(opt):
    # make folder
    save_dir = os.path.join(opt.result_dir, "phantom", "full-dose-rebin-projs")
    os.makedirs(save_dir, exist_ok=True)

    # dataset for testing
    test_dataset = ACRProjSet(opt.acr_dir, desc="full-dose")
    test_loader = t.utils.data.DataLoader(test_dataset)

    # make high-pass filter
    with open(os.path.join(opt.acr_dir, "full-dose-rebin-params.pkl"), "rb") as file:
        rebin_md = load(file)
    hpf = FFTFilter(rebin_md, kernel=opt.recon_filter, device=opt.device)

    # other params
    buffer_size = int(2 * opt.n_frames_mpd - 1)
    mid_idx = int((2 * opt.n_frames_mpd - 1) // 2)

    # temporary buffers
    in_buffer = []
    theta_buffer = []
    zloc_buffer = []
    for t_idx, (_, in_proj, rebin_w, theta, zloc) in tqdm(
        enumerate(test_loader), desc="full-dose-rebin", total=len(test_loader)
    ):
        # copy to device
        in_proj = t.swapaxes(in_proj, 0, 1).to(opt.device)

        rebin_w = rebin_w.squeeze().to(opt.device)

        in_buffer.append(in_proj.squeeze())
        theta_buffer.append(theta.squeeze())
        zloc_buffer.append(zloc.squeeze())

        if len(in_buffer) <= buffer_size:
            continue

        in_buffer = in_buffer[-(buffer_size + 1) :]
        theta_buffer = theta_buffer[-(buffer_size + 1) :]
        zloc_buffer = zloc_buffer[-(buffer_size + 1) :]

        # weighted summation
        rebin_ref = t.sum(rebin_w * t.cat(in_buffer[mid_idx : mid_idx + 2], dim=0), dim=0)

        # filtering
        rebin_ref = hpf.run(rebin_ref)

        # gantry params
        rebin_theta = theta_buffer[mid_idx]
        rebin_zloc = zloc_buffer[mid_idx]

        # save to file
        proj_path = os.path.join(save_dir, "%06d.pkl" % (t_idx - 1 - mid_idx))
        out_dict = {
            "img": rebin_ref.cpu().squeeze().numpy().astype("float32"),
            "theta": rebin_theta.squeeze().numpy().astype("float32"),
            "zloc": rebin_zloc.squeeze().numpy().astype("float32"),
        }

        with open(proj_path, "wb") as file:
            dump(out_dict, file, protocol=HIGHEST_PROTOCOL)


# low-dose projection denoising
def runDenoiseLD(opt):
    # load model
    mpd_net = MPDNet(n_frames=opt.n_frames_mpd, manufacturer=opt.manufacturer).to(opt.device)
    mpd_net.load()
    mpd_net.eval()

    # make folder
    save_dir = os.path.join(opt.result_dir, "phantom", "low-dose-denoise-projs")
    os.makedirs(save_dir, exist_ok=True)

    # dataset for testing
    test_dataset = ACRProjSet(opt.acr_dir, desc="low-dose")
    test_loader = t.utils.data.DataLoader(test_dataset)

    # make high-pass filter
    with open(os.path.join(opt.acr_dir, "low-dose-rebin-params.pkl"), "rb") as file:
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
        enumerate(test_loader), desc="low-dose-denoise", total=len(test_loader)
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
        proj_path = os.path.join(save_dir, "%06d.pkl" % (t_idx - 1 - mid_idx))
        out_dict = {
            "noise": rebin_pred.cpu().squeeze().numpy().astype("float32"),
            "img": rebin_ref.cpu().squeeze().numpy().astype("float32"),
            "theta": rebin_theta.squeeze().numpy().astype("float32"),
            "zloc": rebin_zloc.squeeze().numpy().astype("float32"),
        }

        with open(proj_path, "wb") as file:
            dump(out_dict, file, protocol=HIGHEST_PROTOCOL)


# full-dose image reconstruction
def runReconFD(opt):
    # load metadata
    proj_md = loadProjMetadata(os.path.join(opt.acr_dir, "full-dose-proj-params.pkl"))
    img_md = loadImgMetadata(os.path.join(opt.acr_dir, "full-dose-img-params.pkl"))

    # reconstruction
    data_dir = os.path.join(opt.result_dir, "phantom", "full-dose-rebin-projs")
    save_dir = os.path.join(opt.result_dir, "phantom", "full-dose-ref-imgs")
    os.makedirs(save_dir, exist_ok=True)

    # apply slice insertion
    recon_list = img_md["recon_list"][1:-1]

    # image and projection attributes
    img_hei = int(img_md["Rows"])
    img_wid = int(img_md["Columns"])
    rfov = float(img_md["ReconstructionDiameter"])
    afov = float(img_md["DataCollectionDiameter"])
    thickness = float(img_md["SliceThickness"])
    n_projs_pi = int(proj_md["n_projs_2pi"] / 4)

    # BP variables
    x_grid, y_grid = t.meshgrid(t.arange(0, img_wid), t.arange(0, img_hei), indexing="xy")
    x_grid = ((rfov / img_wid) * (x_grid - (img_wid - 1) / 2)).to(opt.device)
    y_grid = ((rfov / img_hei) * (y_grid - (img_hei - 1) / 2)).to(opt.device)
    bp_grid = t.zeros((1, img_hei, img_wid, 2), device=opt.device)
    out_mask = (x_grid**2 + y_grid**2) <= (afov / 2) ** 2
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
    for s_idx, z_recon in tqdm(enumerate(recon_list), desc="full-dose-recon", total=len(recon_list)):
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
            ref_data = t.tensor(sample["img"], device=opt.device).unsqueeze(0)
            theta = t.tensor(sample["theta"], device=opt.device)
            zloc = t.tensor(sample["zloc"], device=opt.device)

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
                ref_bp = grid_sample(
                    ref_data.unsqueeze(-1), bp_grid, mode="bilinear", padding_mode="border", align_corners=True
                ).squeeze()

                # update weight and BP results
                ref_st = h_t * ref_bp

                p_idx = int(f_idx % n_projs_pi)
                ref_img_s[:, :, p_idx] += t.sum(ref_st, dim=0)
                img_w[:, :, p_idx] += t.sum(h_t, dim=0)

        # weighted average
        ref_img = np.pi * t.mean(ref_img_s / t.clip(img_w, min=1e-8), dim=-1)

        # convert to attenuation ratio
        ref_img = (ref_img - mu_water) / mu_water
        out_img = (1000.0 * ref_img + 1024.0) / 4095.0
        out_img = t.clip(out_img, 0.0, 1.0).squeeze()

        # save to file
        with open(os.path.join(save_dir, "%04d.pkl" % (s_idx + 1)), "wb") as file:
            dump(out_img.cpu().numpy(), file, protocol=HIGHEST_PROTOCOL)

        ref_img = 1000.0 * ref_img
        ref_img = t.clip(ref_img, -1024, 3071).squeeze()
        ref_img = t.rot90(ref_img, -1)

        # save to dicom file
        save_path = os.path.join(save_dir, "%04d.dcm" % (s_idx + 1))
        save2Dicom(ref_img.cpu().numpy(), s_idx + 1, img_md, save_path, desc="full-dose")

    # delete projection data
    rmtree(data_dir)


# low-dose image reconstruction
def runReconLD(opt):
    # load metadata
    proj_md = loadProjMetadata(os.path.join(opt.acr_dir, "low-dose-proj-params.pkl"))
    img_md = loadImgMetadata(os.path.join(opt.acr_dir, "low-dose-img-params.pkl"))

    # reconstruction
    data_dir = os.path.join(opt.result_dir, "phantom", "low-dose-denoise-projs")
    save_dir = os.path.join(opt.result_dir, "phantom", "low-dose-recon-imgs")
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
    afov = float(img_md["DataCollectionDiameter"])
    thickness = float(img_md["SliceThickness"])
    n_projs_pi = int(proj_md["n_projs_2pi"] / 4)

    # BP variables
    x_grid, y_grid = t.meshgrid(t.arange(0, img_wid), t.arange(0, img_hei), indexing="xy")
    x_grid = ((rfov / img_wid) * (x_grid - (img_wid - 1) / 2)).to(opt.device)
    y_grid = ((rfov / img_hei) * (y_grid - (img_hei - 1) / 2)).to(opt.device)
    bp_grid = t.zeros((1, img_hei, img_wid, 2), device=opt.device)
    out_mask = (x_grid**2 + y_grid**2) <= (afov / 2) ** 2
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
    for s_idx, z_recon in tqdm(enumerate(recon_list), desc="low-dose-recon", total=len(recon_list)):
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
            pred_data = t.tensor(sample["noise"], device=opt.device).unsqueeze(0)
            ref_data = t.tensor(sample["img"], device=opt.device).unsqueeze(0)
            theta = t.tensor(sample["theta"], device=opt.device)
            zloc = t.tensor(sample["zloc"], device=opt.device)

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
        out_data = {
            "noise": pred_img.cpu().numpy().astype("float32"),
            "img": ref_img.cpu().numpy().astype("float32"),
            "mask": out_mask.cpu().numpy().astype("bool"),
        }
        with open(img_path, "wb") as file:
            dump(out_data, file, protocol=HIGHEST_PROTOCOL)

    # delete projection data
    rmtree(data_dir)


# image refinement
def runRefineLD(opt):
    # load model
    mir_net = MIRNet(n_frames=opt.n_frames_mir, manufacturer=opt.manufacturer).to(opt.device)
    mir_net.load()
    mir_net.eval()

    # load image metadata
    img_md = loadImgMetadata(os.path.join(opt.acr_dir, "low-dose-img-params.pkl"))

    # make folder
    pred_save_dir = os.path.join(opt.result_dir, "phantom", "low-dose-pred-imgs")
    os.makedirs(pred_save_dir, exist_ok=True)
    ref_save_dir = os.path.join(opt.result_dir, "phantom", "low-dose-ref-imgs")
    os.makedirs(ref_save_dir, exist_ok=True)

    # dataset for testing
    data_dir = os.path.join(opt.result_dir, "phantom", "low-dose-recon-imgs")
    test_dataset = ACRImgSet(data_dir)
    test_loader = t.utils.data.DataLoader(test_dataset)

    # temporary buffers
    noise_buffer = []
    in_buffer = []
    for index, (in_noise, in_img, mask) in tqdm(enumerate(test_loader), desc="low-dose-refine", total=len(test_loader)):
        # copy to device
        in_noise = in_noise.to(opt.device)
        in_img = in_img.to(opt.device)

        noise_buffer.append(in_noise)
        in_buffer.append(in_img)
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

        pred_img = mid_img + pred_noise
        pred_img = (1000.0 * pred_img + 1024.0) / 4095.0
        pred_img = t.clip(pred_img, 0.0, 1.0).squeeze()
        out_img = (1000.0 * mid_img + 1024.0) / 4095.0
        out_img = t.clip(out_img, 0.0, 1.0).squeeze()

        # save to file
        f_idx = (index + 1 - opt.n_frames_mir) // (opt.n_frames_mir // 2) + 1
        with open(os.path.join(pred_save_dir, "%04d.pkl" % f_idx), "wb") as file:
            dump(pred_img.cpu().numpy(), file, protocol=HIGHEST_PROTOCOL)
        with open(os.path.join(ref_save_dir, "%04d.pkl" % f_idx), "wb") as file:
            dump(out_img.cpu().numpy(), file, protocol=HIGHEST_PROTOCOL)

        # post-processing
        pred_img = 1000.0 * (mid_img + pred_noise)
        pred_img = t.clip(pred_img, -1024, 3071).squeeze()
        pred_img = t.rot90(pred_img, -1)

        ref_img = 1000.0 * mid_img
        ref_img = t.clip(ref_img, -1024, 3071).squeeze()
        ref_img = t.rot90(ref_img, -1)

        # save to dicom file
        f_idx = (index + 1 - opt.n_frames_mir) // (opt.n_frames_mir // 2) + 1
        save_path = os.path.join(pred_save_dir, "%04d.dcm" % f_idx)
        save2Dicom(pred_img.cpu().numpy(), f_idx, img_md, save_path, desc="restored")
        save_path = os.path.join(ref_save_dir, "%04d.dcm" % f_idx)
        save2Dicom(ref_img.cpu().numpy(), f_idx, img_md, save_path, desc="low-dose")

    # delete reconstruction data
    rmtree(data_dir)


# process each dosage
def main():
    file_index_list = (
        [i for i in range(13, 24)]
        + [i for i in range(30, 46)]
        + [i for i in range(48, 65)]
        + [i for i in range(70, 81)]
    )

    # load config
    opt = Config(mode="Test", manufacturer="Siemens")

    #  full-dose projections
    runRebinFD(opt)

    # reconstruct full-dose images
    runReconFD(opt)

    # denoise low-dose projections
    runDenoiseLD(opt)

    # reconctruct low-dose images
    runReconLD(opt)

    # refine low-dose images
    runRefineLD(opt)

    # testing
    with open(os.path.join("./preset", "ACR-mask.pkl"), "rb") as file:
        mask = load(file)

    mse_results = []
    ssim_results = []
    for file_idx in file_index_list:
        with open(os.path.join(opt.result_dir, "phantom", "full-dose-ref-imgs", "%04d.pkl" % file_idx), "rb") as f_obj:
            fd_data = load(f_obj)

        with open(os.path.join(opt.result_dir, "phantom", "low-dose-pred-imgs", "%04d.pkl" % file_idx), "rb") as f_obj:
            pred_data = load(f_obj)

        fd_data = fd_data * 4095.0 - 1024.0
        pred_data = pred_data * 4095.0 - 1024.0
        fd_data = np.rot90(fd_data.squeeze(), -1)
        pred_data = np.rot90(pred_data.squeeze(), -1)

        # compute error
        mse_results.append(calcMSE(pred_data, fd_data, 1023.5, 4096, mask))
        ssim_results.append(calcSSIM(pred_data, fd_data, 1023.5, 4096, mask))

    mse_avg = np.mean(mse_results)
    mse_std = np.std(mse_results)
    ssim_avg = np.mean(ssim_results)
    ssim_std = np.std(ssim_results)

    print("Results of phantom, MSE: %.2f±%.2f, SSIM: %.4f±%.4f" % (mse_avg, mse_std, ssim_avg, ssim_std))

    # clean up directories
    folder_path = os.path.join(opt.result_dir, "phantom")
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.endswith(".pkl"):
                file_path = os.path.join(dirpath, filename)
                os.remove(file_path)


if __name__ == "__main__":
    main()
