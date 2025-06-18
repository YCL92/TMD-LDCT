import os
from pickle import HIGHEST_PROTOCOL, dump, load

import numpy as np
import torch as t
from pydicom import dcmread
from skimage.transform.radon_transform import _get_fourier_filter
from torch.nn.functional import grid_sample, pad, relu
from tqdm import tqdm


# alternative 2D interpolation
class Interp2:

    def __init__(self, grid, offset=[0, 0], data_size=None):
        self.p_max, self.r_max = data_size

        interp_p = t.clamp(grid[:, 0] + offset[0], 0, self.p_max - 1)
        interp_r = t.clamp(grid[:, 1] + offset[1], 0, self.r_max - 1)
        int_p = t.floor(interp_p)
        int_r = t.floor(interp_r)

        w_p = interp_p - int_p
        w_r = interp_r - int_r

        self.w00 = (1.0 - w_p) * (1.0 - w_r)
        self.w01 = (1.0 - w_p) * w_r
        self.w10 = w_p * (1.0 - w_r)
        self.w11 = w_p * w_r

        self.p_idx = int_p.long()
        self.r_idx = int_r.long()

    def getWeights(self):
        return t.stack([self.w00, self.w10, self.w01, self.w11], dim=0)

    def run(self, data):
        n00 = data[:, self.p_idx, self.r_idx]
        n10 = data[:, t.clamp(self.p_idx + 1, 0, self.p_max - 1), self.r_idx]

        return t.stack([n00, n10], dim=0)


# frequency domain filtering
class FFTFilter:

    def __init__(self, proj_md, kernel="shepp-logan", device="cpu"):
        f_len = proj_md["shape"][-1]
        self.h_filter = _get_fourier_filter(f_len, kernel).astype("float32")
        self.h_filter = t.tensor(self.h_filter, device=device).squeeze().view(1, 1, -1)
        self.h_filter = t.fft.fftshift(self.h_filter / (2 * proj_md["ds"]))

    def run(self, proj):
        n_pad = int(proj.size(-1) / 2)
        padded_data = pad(proj, (n_pad, n_pad), mode="constant", value=0)
        padded_filter = pad(self.h_filter, (n_pad, n_pad), mode="constant", value=0)

        fft_data = t.fft.fftshift(t.fft.fft(padded_data, dim=-1))
        fft_data = fft_data * padded_filter
        filtered_data = t.real(t.fft.ifft(t.fft.ifftshift(fft_data), dim=-1))
        filtered_data = filtered_data[..., n_pad:-n_pad]

        return filtered_data


# rebinning (without flying focal spots)
def rebin(data_dir, save_dir, proj_md, device="cpu", desc=""):
    # make folder
    os.makedirs(save_dir, exist_ok=True)

    # load gantry params
    n_projs = proj_md["n_projs_full"]
    n_rows = proj_md["n_rows"]
    n_chnls = proj_md["n_chnls"]
    proj_offset = proj_md["proj_offset"]
    proj_offset_full = int(2 * proj_offset)

    # interpolation grids
    grid_offsets = [0, proj_offset]
    fs_grid = t.zeros((2 * n_chnls, 2), device=device)
    fs_grid[:, 0] = (n_chnls - 1) - proj_md["p_idx"]
    fs_grid[:, 1] = proj_md["dr_idx"]

    # pre-compute interpolation weight
    interp_fs = Interp2(fs_grid, offset=grid_offsets, data_size=[n_chnls, proj_offset_full + 1])
    save_dict = {
        "manufacturer": proj_md["manufacturer"],
        "ds": proj_md["ds"],
        "shape": (n_rows, int(2 * n_chnls)),
        "fs_w": interp_fs.getWeights().cpu().numpy().astype("float32"),
    }
    with open(os.path.join(save_dir, "../", "rebin-params.pkl"), "wb") as file:
        dump(save_dict, file, protocol=HIGHEST_PROTOCOL)

    # temporary buffers
    ld_data = t.zeros((n_rows, n_chnls, proj_offset_full + 1), device=device)
    ld_noise = t.zeros((1, n_chnls, proj_offset_full + 1), device=device)
    fd_data = t.zeros((n_rows, n_chnls, proj_offset_full + 1), device=device)
    fd_noise = t.zeros((1, n_chnls, proj_offset_full + 1), device=device)

    # pointers for acceleration purpose
    for r_idx in tqdm(range(n_projs), desc=desc, total=n_projs):
        # update buffers
        ld_data = t.roll(ld_data, -1, dims=-1)
        ld_noise = t.roll(ld_noise, -1, dims=-1)
        fd_data = t.roll(fd_data, -1, dims=-1)
        fd_noise = t.roll(fd_noise, -1, dims=-1)

        # load low-dose data
        file = dcmread(os.path.join(data_dir, "low-dose-projs", "%06d.dcm" % r_idx))
        slope = float(file.RescaleSlope)
        intercept = float(file.RescaleIntercept)
        ld_data[:, :, -1] = (
            slope * t.tensor(np.flipud(file.pixel_array.transpose()).astype("float32"), device=device) + intercept
        )
        ld_noise[:, :, -1] = t.tensor(np.frombuffer(file[0x7033, 0x1065].value, dtype="float32"))

        # load full-dose data
        file = dcmread(os.path.join(data_dir, "full-dose-projs", "%06d.dcm" % r_idx))
        slope = float(file.RescaleSlope)
        intercept = float(file.RescaleIntercept)
        fd_data[:, :, -1] = (
            slope * t.tensor(np.flipud(file.pixel_array.transpose()).astype("float32"), device=device) + intercept
        )
        fd_noise[:, :, -1] = t.tensor(np.frombuffer(file[0x7033, 0x1065].value, dtype="float32"))

        # loop until sufficient data collected
        if r_idx < proj_offset_full:
            continue

        # rebinning
        rebin_ld_data = interp_fs.run(ld_data).squeeze()
        rebin_ld_noise = interp_fs.run(ld_noise).squeeze()
        rebin_fd_data = interp_fs.run(fd_data).squeeze()
        rebin_fd_noise = interp_fs.run(fd_noise).squeeze()

        # read other params
        file = dcmread(os.path.join(data_dir, "full-dose-projs", "%06d.dcm" % (r_idx - proj_offset)))
        rebin_theta = np.frombuffer(file[0x7031, 0x1001].value, dtype="float32").item()
        rebin_zloc = np.frombuffer(file[0x7031, 0x1002].value, dtype="float32").item()

        # write to file
        proj_path = os.path.join(save_dir, "%06d.pkl" % r_idx)
        out_data = {
            "rebin_ld_data": rebin_ld_data.cpu().numpy().astype("float16"),
            "rebin_ld_noise": rebin_ld_noise.cpu().numpy().astype("float32"),
            "rebin_fd_data": rebin_fd_data.cpu().numpy().astype("float16"),
            "rebin_fd_noise": rebin_fd_noise.cpu().numpy().astype("float32"),
            "rebin_theta": rebin_theta,
            "rebin_zloc": rebin_zloc,
        }
        with open(proj_path, "wb") as file:
            dump(out_data, file, protocol=HIGHEST_PROTOCOL)


# rebinning (with flying focal spots)
def rebinFFS(data_dir, save_dir, proj_md, device="cpu", desc=""):
    # make folder
    os.makedirs(save_dir, exist_ok=True)

    # load gantry params
    n_projs = int(proj_md["n_projs_full"] // 2)
    n_rows = proj_md["n_rows"]
    n_chnls = proj_md["n_chnls"]
    proj_offset = proj_md["proj_offset"]
    proj_offset_full = int(2 * proj_offset)

    # interpolation grids
    grid_offsets = [0, proj_offset]
    fs0_grid = t.zeros((2 * n_chnls, 2), device=device)
    fs1_grid = t.zeros((2 * n_chnls, 2), device=device)
    fs0_grid[:, 0] = (n_chnls - 1) - proj_md["p_idx0"]
    fs0_grid[:, 1] = proj_md["dr_idx0"]
    fs1_grid[:, 0] = (n_chnls - 1) - proj_md["p_idx1"]
    fs1_grid[:, 1] = proj_md["dr_idx1"]

    # pre-compute weight maps
    interp_fs0 = Interp2(fs0_grid, offset=grid_offsets, data_size=[n_chnls, proj_offset_full + 1])
    interp_fs1 = Interp2(fs1_grid, offset=grid_offsets, data_size=[n_chnls, proj_offset_full + 1])

    save_dict = {
        "manufacturer": proj_md["manufacturer"],
        "ds": proj_md["ds"],
        "shape": (n_rows, int(2 * n_chnls)),
        "fs0_w": interp_fs0.getWeights().cpu().numpy().astype("float32"),
        "fs1_w": interp_fs1.getWeights().cpu().numpy().astype("float32"),
    }
    with open(os.path.join(save_dir, "../", "rebin-params.pkl"), "wb") as file:
        dump(save_dict, file, protocol=HIGHEST_PROTOCOL)

    # temporary buffers
    ld_data0 = t.zeros((n_rows, n_chnls, proj_offset_full + 1), device=device)
    ld_data1 = t.zeros((n_rows, n_chnls, proj_offset_full + 1), device=device)
    ld_noise0 = t.zeros((1, n_chnls, proj_offset_full + 1), device=device)
    ld_noise1 = t.zeros((1, n_chnls, proj_offset_full + 1), device=device)
    fd_data0 = t.zeros((n_rows, n_chnls, proj_offset_full + 1), device=device)
    fd_data1 = t.zeros((n_rows, n_chnls, proj_offset_full + 1), device=device)
    fd_noise0 = t.zeros((1, n_chnls, proj_offset_full + 1), device=device)
    fd_noise1 = t.zeros((1, n_chnls, proj_offset_full + 1), device=device)

    # make folder
    os.makedirs(save_dir, exist_ok=True)

    # pointers for acceleration purpose
    for r_idx in tqdm(range(n_projs), desc=desc, total=n_projs):
        # update buffers
        ld_data0 = t.roll(ld_data0, -1, dims=-1)
        ld_data1 = t.roll(ld_data1, -1, dims=-1)
        ld_noise0 = t.roll(ld_noise0, -1, dims=-1)
        ld_noise1 = t.roll(ld_noise1, -1, dims=-1)
        fd_data0 = t.roll(fd_data0, -1, dims=-1)
        fd_data1 = t.roll(fd_data1, -1, dims=-1)
        fd_noise0 = t.roll(fd_noise0, -1, dims=-1)
        fd_noise1 = t.roll(fd_noise1, -1, dims=-1)

        # load low-dose data
        file = dcmread(os.path.join(data_dir, "low-dose-projs", "fs0", "%06d.dcm" % r_idx))
        slope = float(file.RescaleSlope)
        intercept = float(file.RescaleIntercept)
        ld_data0[:, :, -1] = slope * t.tensor(file.pixel_array.transpose().astype("float32"), device=device) + intercept
        ld_noise0[:, :, -1] = t.tensor(np.frombuffer(file[0x7033, 0x1065].value, dtype="float32"))

        file = dcmread(os.path.join(data_dir, "low-dose-projs", "fs1", "%06d.dcm" % r_idx))
        slope = float(file.RescaleSlope)
        intercept = float(file.RescaleIntercept)
        ld_data1[:, :, -1] = slope * t.tensor(file.pixel_array.transpose().astype("float32"), device=device) + intercept
        ld_noise1[:, :, -1] = t.tensor(np.frombuffer(file[0x7033, 0x1065].value, dtype="float32"))

        # load full-dose data
        file = dcmread(os.path.join(data_dir, "full-dose-projs", "fs0", "%06d.dcm" % r_idx))
        slope = float(file.RescaleSlope)
        intercept = float(file.RescaleIntercept)
        fd_data0[:, :, -1] = slope * t.tensor(file.pixel_array.transpose().astype("float32"), device=device) + intercept
        fd_noise0[:, :, -1] = t.tensor(np.frombuffer(file[0x7033, 0x1065].value, dtype="float32"))

        file = dcmread(os.path.join(data_dir, "full-dose-projs", "fs1", "%06d.dcm" % r_idx))
        slope = float(file.RescaleSlope)
        intercept = float(file.RescaleIntercept)
        fd_data1[:, :, -1] = slope * t.tensor(file.pixel_array.transpose().astype("float32"), device=device) + intercept
        fd_noise1[:, :, -1] = t.tensor(np.frombuffer(file[0x7033, 0x1065].value, dtype="float32"))

        # loop until sufficient data collected
        if r_idx < proj_offset_full:
            continue

        # rebinning
        rebin_ld_data0 = interp_fs0.run(ld_data0).squeeze()
        rebin_ld_noise0 = interp_fs0.run(ld_noise0).squeeze()
        rebin_ld_data1 = interp_fs1.run(ld_data1).squeeze()
        rebin_ld_noise1 = interp_fs1.run(ld_noise1).squeeze()

        rebin_fd_data0 = interp_fs0.run(fd_data0).squeeze()
        rebin_fd_noise0 = interp_fs0.run(fd_noise0).squeeze()
        rebin_fd_data1 = interp_fs1.run(fd_data1).squeeze()
        rebin_fd_noise1 = interp_fs1.run(fd_noise1).squeeze()

        # read other params
        file = dcmread(os.path.join(data_dir, "full-dose-projs", "fs0", "%06d.dcm" % (r_idx - proj_offset)))
        rebin_theta = np.frombuffer(file[0x7031, 0x1001].value, dtype="float32").item()
        rebin_zloc = -np.frombuffer(file[0x7031, 0x1002].value, dtype="float32").item()  # reverse z coordinates

        # write to file
        proj_path = os.path.join(save_dir, "%06d.pkl" % r_idx)
        out_data = {
            "rebin_ld_data0": rebin_ld_data0.cpu().numpy().astype("float16"),
            "rebin_ld_data1": rebin_ld_data1.cpu().numpy().astype("float16"),
            "rebin_ld_noise0": rebin_ld_noise0.cpu().numpy().astype("float32"),
            "rebin_ld_noise1": rebin_ld_noise1.cpu().numpy().astype("float32"),
            "rebin_fd_data0": rebin_fd_data0.cpu().numpy().astype("float16"),
            "rebin_fd_data1": rebin_fd_data1.cpu().numpy().astype("float16"),
            "rebin_fd_noise0": rebin_fd_noise0.cpu().numpy().astype("float32"),
            "rebin_fd_noise1": rebin_fd_noise1.cpu().numpy().astype("float32"),
            "rebin_theta": rebin_theta,
            "rebin_zloc": rebin_zloc,
        }

        with open(proj_path, "wb") as file:
            dump(out_data, file, protocol=HIGHEST_PROTOCOL)


# back-projection
def recon(data_dir, save_dir, proj_md, img_md, n_interp=None, device="cpu", desc=""):
    # make folder
    os.makedirs(save_dir, exist_ok=True)

    # apply slice insertion
    if n_interp is not None:
        recon_list = []
        for i_idx in range(len(img_md["recon_list"]) - 1):
            interval = (img_md["recon_list"][i_idx + 1] - img_md["recon_list"][i_idx]) / n_interp
            for j_idx in range(n_interp):
                recon_list.append(img_md["recon_list"][i_idx] + j_idx * interval)
        recon_list.append(img_md["recon_list"][-1])
    else:
        recon_list = img_md["recon_list"]

    # image and projection attributes
    img_hei = int(img_md["Rows"])
    img_wid = int(img_md["Columns"])
    rfov = float(img_md["ReconstructionDiameter"])
    afov = float(img_md["DataCollectionDiameter"])
    thickness = float(img_md["SliceThickness"])

    if proj_md["manufacturer"] == "Siemens":
        n_projs_pi = int(proj_md["n_projs_2pi"] / 4)
    elif proj_md["manufacturer"] == "GE":
        n_projs_pi = int(proj_md["n_projs_2pi"] / 2)
    else:
        raise NotImplementedError()

    # BP variables
    x_grid, y_grid = t.meshgrid(t.arange(0, img_wid), t.arange(0, img_hei), indexing="xy")
    x_grid = ((rfov / img_wid) * (x_grid - (img_wid - 1) / 2)).to(device)
    y_grid = ((rfov / img_hei) * (y_grid - (img_hei - 1) / 2)).to(device)
    bp_grid = t.zeros((1, img_hei, img_wid, 2), device=device)
    out_mask = (x_grid**2 + y_grid**2) <= (afov / 2) ** 2
    rebin_qtan = proj_md["rebin_qtan"].to(device)
    rebin_wq = proj_md["rebin_wq"].to(device)
    mu_water = proj_md["mu_water"].to(device)

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
    for s_idx, z_recon in tqdm(enumerate(recon_list), desc=desc, total=len(recon_list)):
        # temporary buffers
        pred_img_s = t.zeros((img_hei, img_wid, n_projs_pi), device=device)
        ld_img_s = t.zeros((img_hei, img_wid, n_projs_pi), device=device)
        fd_img_s = t.zeros((img_hei, img_wid, n_projs_pi), device=device)
        img_w = t.zeros((img_hei, img_wid, n_projs_pi), device=device)

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
            pred_data = sample["noise"].to(device).squeeze().unsqueeze(0)
            ld_data = sample["ld"].to(device).squeeze().unsqueeze(0)
            fd_data = sample["fd"].to(device).squeeze().unsqueeze(0)
            theta = sample["theta"].to(device).squeeze()
            zloc = sample["zloc"].to(device).squeeze()

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
                ld_bp = grid_sample(
                    ld_data.unsqueeze(-1), bp_grid, mode="bilinear", padding_mode="border", align_corners=True
                ).squeeze()
                fd_bp = grid_sample(
                    fd_data.unsqueeze(-1), bp_grid, mode="bilinear", padding_mode="border", align_corners=True
                ).squeeze()

                # update weight and BP results
                pred_st = h_t * pred_bp
                ld_st = h_t * ld_bp
                fd_st = h_t * fd_bp

                p_idx = int(f_idx % n_projs_pi)
                pred_img_s[:, :, p_idx] += t.sum(pred_st, dim=0)
                ld_img_s[:, :, p_idx] += t.sum(ld_st, dim=0)
                fd_img_s[:, :, p_idx] += t.sum(fd_st, dim=0)
                img_w[:, :, p_idx] += t.sum(h_t, dim=0)

        # weighted average
        pred_img = np.pi * t.mean(pred_img_s / t.clip(img_w, min=1e-8), dim=-1)
        ld_img = np.pi * t.mean(ld_img_s / t.clip(img_w, min=1e-8), dim=-1)
        fd_img = np.pi * t.mean(fd_img_s / t.clip(img_w, min=1e-8), dim=-1)

        # convert to attenuation ratio
        pred_img = pred_img / mu_water
        ld_img = (ld_img - mu_water) / mu_water
        fd_img = (fd_img - mu_water) / mu_water

        # write to file
        img_path = os.path.join(save_dir, "%04d.pkl" % s_idx)
        out_data = {
            "noise": pred_img.cpu().numpy(),
            "ld": ld_img.cpu().numpy(),
            "fd": fd_img.cpu().numpy(),
            "mask": out_mask.cpu().numpy(),
        }

        with open(img_path, "wb") as file:
            dump(out_data, file, protocol=HIGHEST_PROTOCOL)
