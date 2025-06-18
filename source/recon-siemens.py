import os
import shutil
from pickle import load, dump, HIGHEST_PROTOCOL

import numpy as np
import torch as t
from pydicom import dcmread
from skimage.transform.radon_transform import _get_fourier_filter
from torch.nn.functional import pad, grid_sample, relu
from tqdm import tqdm

from preset.config import Config
from util.dicomio import loadProjMetadata, loadImgMetadata, save2Dicom


# filtering
def fftFilter(proj, h_filter):
    n_pad = int(proj.size(-1) / 2)
    padded_data = pad(proj, (n_pad, n_pad), mode="constant", value=0)
    padded_filter = pad(h_filter, (n_pad, n_pad), mode="constant", value=0)

    fft_data = t.fft.fftshift(t.fft.fft(padded_data, dim=-1))
    fft_data = fft_data * padded_filter
    filtered_data = t.real(t.fft.ifft(t.fft.ifftshift(fft_data), dim=-1))
    filtered_data = filtered_data[..., n_pad:-n_pad]

    return filtered_data


# rebin then apply filter
def rebinNFilter(data_dir, save_dir, proj_md, flt_kernel, device):
    # projection attributes
    n_projs = int(proj_md["n_projs_full"] // 2)
    n_rebin_rows = int(2 * proj_md["n_rows"])
    n_rebin_chnls = int(2 * proj_md["n_chnls"])
    proj_offset_full = int(2 * proj_md["proj_offset"])

    # make filter kernel
    # experimental, half width due to double sampling, will be padded to full length
    h_filter = (
        t.tensor(_get_fourier_filter(n_rebin_chnls, flt_kernel).astype("float32"), device=device)
        .squeeze()
        .view(1, 1, -1)
    )
    h_filter = t.fft.fftshift(h_filter / (2 * proj_md["ds"]))

    # interpolation grids
    fs0_grid = t.zeros((1, n_rebin_chnls, 1, 2), device=device)
    fs1_grid = t.zeros((1, n_rebin_chnls, 1, 2), device=device)
    fs0_grid[0, :, 0, 0] = proj_md["dr_idx0"] / proj_md["proj_offset"]
    fs0_grid[0, :, 0, 1] = 1 - 2 * proj_md["p_idx0"] / (proj_md["n_chnls"] - 1)
    fs1_grid[0, :, 0, 0] = proj_md["dr_idx1"] / proj_md["proj_offset"]
    fs1_grid[0, :, 0, 1] = 1 - 2 * proj_md["p_idx1"] / (proj_md["n_chnls"] - 1)

    # temporary buffers
    raw_data0 = t.zeros((1, proj_md["n_rows"], proj_md["n_chnls"], proj_offset_full + 1), device=device)
    raw_data1 = t.zeros((1, proj_md["n_rows"], proj_md["n_chnls"], proj_offset_full + 1), device=device)
    rebin_data = t.zeros((1, n_rebin_rows, n_rebin_chnls), device=device)

    # variables to be saved
    param_list = []

    # make folder
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)

    # pointers for acceleration purpose
    for r_idx in tqdm(range(n_projs), desc="Rebin", total=n_projs):
        # grab data
        raw_data0 = t.roll(raw_data0, -1, dims=-1)
        raw_data1 = t.roll(raw_data1, -1, dims=-1)

        file = dcmread(os.path.join(data_dir, "fs0", "%06d.dcm" % r_idx))
        slope = float(file.RescaleSlope)
        intercept = float(file.RescaleIntercept)
        raw_data0[0, :, :, -1] = (
            slope * t.tensor(file.pixel_array.transpose().astype("float32"), device=device) + intercept
        )

        file = dcmread(os.path.join(data_dir, "fs1", "%06d.dcm" % r_idx))
        slope = float(file.RescaleSlope)
        intercept = float(file.RescaleIntercept)
        raw_data1[0, :, :, -1] = (
            slope * t.tensor(file.pixel_array.transpose().astype("float32"), device=device) + intercept
        )

        # loop until sufficient data collected
        if r_idx < proj_offset_full:
            continue

        # focal spot 0 rebining
        rebin_data[0, 0::2, :] = grid_sample(
            raw_data0, fs0_grid, mode="bilinear", padding_mode="border", align_corners=True
        ).squeeze()

        # focal spot 1 rebining
        rebin_data[0, 1::2, :] = grid_sample(
            raw_data1, fs1_grid, mode="bilinear", padding_mode="border", align_corners=True
        ).squeeze()

        # filtering
        rebin_data = fftFilter(rebin_data, h_filter)

        # extra metadata
        file = dcmread(os.path.join(data_dir, "fs0", "%06d.dcm" % (r_idx - proj_md["proj_offset"])))
        rebin_theta = np.frombuffer(file[0x7031, 0x1001].value, dtype="float32").item()
        rebin_zloc = -np.frombuffer(file[0x7031, 0x1002].value, dtype="float32").item()  # reverse z coordinates
        f_name = "%06d.pkl" % r_idx
        param_list.append((f_name, rebin_theta, rebin_zloc))

        # save projection to file
        with open(os.path.join(save_dir, "%06d.pkl" % r_idx), "wb") as file:
            dump(rebin_data.cpu().numpy(), file, protocol=HIGHEST_PROTOCOL)

    # save other params to file
    with open(os.path.join(save_dir, "params.pkl"), "wb") as file:
        dump(param_list, file, protocol=HIGHEST_PROTOCOL)


# back-projection
def backProjRecon(data_dir, save_dir, proj_md, img_md, device, desc=""):
    # image and projection attributes
    img_hei = int(img_md["Rows"])
    img_wid = int(img_md["Columns"])
    rfov = float(img_md["ReconstructionDiameter"])
    afov = float(img_md["DataCollectionDiameter"])
    thickness = float(img_md["SliceThickness"])
    n_projs_pi = int(proj_md["n_projs_2pi"] / 4)

    # BP variables
    x_grid, y_grid = t.meshgrid(t.arange(0, img_wid), t.arange(0, img_hei), indexing="xy")
    x_grid = ((rfov / img_wid) * (x_grid - (img_wid - 1) / 2)).to(device)
    y_grid = ((rfov / img_hei) * (y_grid - (img_hei - 1) / 2)).to(device)
    bp_grid = t.zeros((1, img_hei, img_wid, 2), device=device)
    out_mask = (x_grid**2 + y_grid**2) > (afov / 2) ** 2
    rebin_qtan = proj_md["rebin_qtan"].to(device)
    rebin_wq = proj_md["rebin_wq"].to(device)

    # expand to 3D without taking extra memory
    x_grid = x_grid.unsqueeze(0)
    y_grid = y_grid.unsqueeze(0)
    rebin_qtan = rebin_qtan.view(-1, 1, 1).expand(-1, img_hei, img_wid)
    rebin_wq = rebin_wq.view(-1, 1, 1).expand(-1, img_hei, img_wid)

    # make folder
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir, exist_ok=True)

    # load file list, angles, and z locations
    with open(os.path.join(data_dir, "params.pkl"), "rb") as file:
        param_list = load(file)

    # back-projection
    start_idx = 0
    end_idx = len(param_list)
    for img_idx, z_recon in tqdm(enumerate(img_md["recon_list"]), desc="Recon", total=len(img_md["recon_list"])):
        # temporary buffers
        img_s = t.zeros((img_hei, img_wid, n_projs_pi), device=device)
        img_w = t.zeros((img_hei, img_wid, n_projs_pi), device=device)

        # loop over all rebined projections
        entry_flag = False
        for f_idx, (f_name, theta, zloc) in enumerate(param_list):
            # make sure file is in the grabing range
            if f_idx < start_idx:
                continue
            elif f_idx > end_idx:
                break

            # calculate distances, weight, and BP results
            p_hat = x_grid * np.sin(theta) - y_grid * np.cos(theta)
            l_hat = t.sqrt(proj_md["r_f"] ** 2 - p_hat**2) - x_grid * np.cos(theta) - y_grid * np.sin(theta)
            dz = zloc - z_recon - proj_md["z_rot"] / (2 * np.pi) * t.arcsin(p_hat / proj_md["r_f"]) + l_hat * rebin_qtan
            h_t = relu(1 - t.abs(dz) / thickness) * rebin_wq

            if t.sum(h_t[:]) > 0:
                if entry_flag is False:
                    entry_flag = True
                    start_idx = max(0, f_idx - n_projs_pi)
                else:
                    end_idx = min(f_idx + n_projs_pi, len(param_list) - 1)

                # load projection data
                with open(os.path.join(data_dir, f_name), "rb") as file:
                    proj_data = load(file)
                rebin_data = t.tensor(proj_data, device=device)

                # 1D interpolation
                bp_grid[0, :, :, 0] = 0
                bp_grid[0, :, :, 1] = p_hat / proj_md["rebin_pmax"]
                bp_data = grid_sample(
                    rebin_data.unsqueeze(-1), bp_grid, mode="bilinear", padding_mode="border", align_corners=True
                ).squeeze()

                # update weight and BP results
                s_t = h_t * bp_data
                p_idx = int(f_idx % n_projs_pi)
                img_s[:, :, p_idx] = img_s[:, :, p_idx] + t.sum(s_t, dim=0)
                img_w[:, :, p_idx] = img_w[:, :, p_idx] + t.sum(h_t, dim=0)

        # sum all BP results and covert to Hounsfield Units
        out_img = np.pi * t.mean(img_s / t.clip(img_w, min=1e-8), dim=-1)
        out_img = 1000 * ((out_img - proj_md["mu_water"]) / proj_md["mu_water"])
        out_img = t.clip(out_img, min=-1024, max=3071)
        out_img[out_mask] = -1024  # remove invalid voxels
        out_img = t.rot90(out_img, -1)

        # save image to dicom file
        img_path = os.path.join(save_dir, "%04d.dcm" % img_idx)
        save2Dicom(out_img.cpu().numpy(), img_idx, img_md, img_path, desc=desc)


# load basic config
opt = Config(manufacturer="Siemens")
study_list = opt.test_list
study_list.sort()

# process each study
for study in study_list:
    print("Processing %s..." % study)

    proj_md_path = os.path.join(opt.proj_dir, study, "proj-params.pkl")
    img_md_path = os.path.join(opt.proj_dir, study, "img-params.pkl")

    # load metadata
    proj_md = loadProjMetadata(proj_md_path)
    img_md = loadImgMetadata(img_md_path)

    # RECONSTRUCTION STARTS HERE
    for series in ["full-dose", "low-dose"]:
        # rebin and apply filtering
        data_dir = os.path.join(opt.proj_dir, study, series + "-projs")
        save_dir = os.path.join(opt.result_dir, study, series + "-temp")
        rebinNFilter(data_dir, save_dir, proj_md, opt.recon_filter, opt.device)

        # back-projection
        data_dir = os.path.join(opt.result_dir, study, series + "-temp")
        save_dir = os.path.join(opt.result_dir, study, series + "-imgs")
        backProjRecon(data_dir, save_dir, proj_md, img_md, opt.device, desc=series)

        # remove temporary files
        shutil.rmtree(os.path.join(opt.result_dir, study, series + "-temp"))
