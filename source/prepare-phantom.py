import os
import zipfile
from pickle import HIGHEST_PROTOCOL, dump
from shutil import move, rmtree

import numpy as np
import torch as t
from pydicom import dcmread
from tqdm import tqdm

from preset.config import Config
from util.dicomio import getImgMeta, getSiemensProjMeta, loadProjMetadata
from util.reconutil import Interp2


# rebinning for phantom scans (with flying focal spots)
def rebinACR(data_dir, proj_md, device="cpu", desc=""):
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
    params = {
        "manufacturer": "Siemens",
        "ds": proj_md["ds"].cpu().numpy().astype("float32"),
        "shape": (n_rows, int(2 * n_chnls)),
        "fs0_w": interp_fs0.getWeights().cpu().numpy().astype("float32"),
        "fs1_w": interp_fs1.getWeights().cpu().numpy().astype("float32"),
    }

    # write to file
    save_path = os.path.join(data_dir, "../", desc + "-rebin-params.pkl")
    with open(save_path, "wb") as file:
        dump(params, file, protocol=HIGHEST_PROTOCOL)

    # temporary buffers
    raw_data0 = t.zeros((n_rows, n_chnls, proj_offset_full + 1), device=device)
    raw_data1 = t.zeros((n_rows, n_chnls, proj_offset_full + 1), device=device)
    raw_noise0 = t.zeros((1, n_chnls, proj_offset_full + 1), device=device)
    raw_noise1 = t.zeros((1, n_chnls, proj_offset_full + 1), device=device)

    # make folder for results
    save_dir = os.path.join(data_dir, "../", desc + "-rebin-projs")
    os.makedirs(save_dir, exist_ok=True)

    # pointers for acceleration purpose
    for r_idx in tqdm(range(n_projs), desc=desc, total=n_projs):
        # grab data
        raw_data0 = t.roll(raw_data0, -1, dims=-1)
        raw_data1 = t.roll(raw_data1, -1, dims=-1)
        raw_noise0 = t.roll(raw_noise0, -1, dims=-1)
        raw_noise1 = t.roll(raw_noise1, -1, dims=-1)

        file = dcmread(os.path.join(data_dir, "fs0", "%06d.dcm" % r_idx))
        slope = float(file.RescaleSlope)
        intercept = float(file.RescaleIntercept)
        raw_data0[:, :, -1] = (
            slope * t.tensor(file.pixel_array.transpose().astype("float32"), device=device) + intercept
        )
        raw_noise0[:, :, -1] = t.tensor(np.frombuffer(file[0x7033, 0x1065].value, dtype="float32"))

        file = dcmread(os.path.join(data_dir, "fs1", "%06d.dcm" % r_idx))
        slope = float(file.RescaleSlope)
        intercept = float(file.RescaleIntercept)
        raw_data1[:, :, -1] = (
            slope * t.tensor(file.pixel_array.transpose().astype("float32"), device=device) + intercept
        )
        raw_noise1[:, :, -1] = t.tensor(np.frombuffer(file[0x7033, 0x1065].value, dtype="float32"))

        # loop until sufficient data collected
        if r_idx < proj_offset_full:
            continue

        # rebinning
        rebin_data0 = interp_fs0.run(raw_data0).squeeze()
        rebin_noise0 = interp_fs0.run(raw_noise0).squeeze()
        rebin_data1 = interp_fs1.run(raw_data1).squeeze()
        rebin_noise1 = interp_fs1.run(raw_noise1).squeeze()

        # read other params
        file = dcmread(os.path.join(data_dir, "fs0", "%06d.dcm" % (r_idx - proj_offset)))
        rebin_theta = np.frombuffer(file[0x7031, 0x1001].value, dtype="float32").item()
        rebin_zloc = -np.frombuffer(file[0x7031, 0x1002].value, dtype="float32").item()  # reversed z coordinate

        save_data = {
            "rebin_data0": rebin_data0.cpu().numpy().astype("float16"),
            "rebin_data1": rebin_data1.cpu().numpy().astype("float16"),
            "rebin_noise0": rebin_noise0.cpu().numpy().astype("float32"),
            "rebin_noise1": rebin_noise1.cpu().numpy().astype("float32"),
            "rebin_theta": rebin_theta,
            "rebin_zloc": rebin_zloc,
        }

        # write to file
        with open(os.path.join(save_dir, "%06d.pkl" % r_idx), "wb") as file:
            dump(save_data, file, protocol=HIGHEST_PROTOCOL)


def main():
    # load config
    opt = Config()

    # find all folders
    folder_list = [item for item in os.listdir(opt.acr_dir) if os.path.isdir(os.path.join(opt.acr_dir, item))]
    for folder in folder_list:
        folder_path = os.path.join(opt.acr_dir, folder)

        if "projection" in folder.lower():
            zip_list = [item for item in os.listdir(folder_path) if ".zip" in item]

            # unzip files
            for zip_file in zip_list:

                zip_file_path = os.path.join(folder_path, zip_file)
                with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                    zip_ref.extractall(opt.acr_dir)

                if "fd" in zip_file.lower():
                    dst_dir = os.path.join(opt.acr_dir, "full-dose-projs")
                    src_dir = os.path.join(opt.acr_dir, zip_file.split("_")[0] + "_FD")
                else:
                    src_dir = os.path.join(opt.acr_dir, zip_file.split("_")[0] + "_LD")
                    dst_dir = os.path.join(opt.acr_dir, "low-dose-projs")
                os.rename(src_dir, dst_dir)

        elif "image" in folder.lower():
            img_dir = os.path.join(folder_path, "3mm B30")
            zip_list = [item for item in os.listdir(img_dir) if ".zip" in item]

            # unzip files
            for zip_file in zip_list:
                zip_file_path = os.path.join(img_dir, zip_file)
                with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                    zip_ref.extractall(opt.acr_dir)

                if "full" in zip_file.lower():
                    dst_dir = os.path.join(opt.acr_dir, "full-dose-imgs")
                    src_dir = os.path.join(opt.acr_dir, "200EFFMAS_3_0_B30F_0005")
                else:
                    dst_dir = os.path.join(opt.acr_dir, "low-dose-imgs")
                    src_dir = os.path.join(opt.acr_dir, "quarter_3mm")
                os.rename(src_dir, dst_dir)

        # delete folder
        rmtree(folder_path)

    # extract projection metadata
    for dosage in ["full-dose", "low-dose"]:
        # extract image metadata
        fd_dir = os.path.join(opt.acr_dir, dosage + "-imgs")
        file_list = [item for item in os.listdir(fd_dir) if ".IMA" in item]
        file_list.sort()

        img_meta = getImgMeta(fd_dir, file_list)

        save_path = os.path.join(opt.acr_dir, dosage + "-img-params.pkl")
        with open(save_path, "wb") as file:
            dump(img_meta, file, protocol=HIGHEST_PROTOCOL)

        # delete folder
        rmtree(os.path.join(opt.acr_dir, dosage + "-imgs"))

        # extract projection metadata and move files to new folders
        folder_path = os.path.join(opt.acr_dir, dosage + "-projs")
        file_list = [item for item in os.listdir(folder_path) if item.endswith(".dcm")]
        file_list.sort()

        proj_meta = getSiemensProjMeta(folder_path, file_list)
        save_path = os.path.join(opt.acr_dir, dosage + "-proj-params.pkl")
        with open(save_path, "wb") as file:
            dump(proj_meta, file, protocol=HIGHEST_PROTOCOL)

        # rename and move to new folders
        os.makedirs(os.path.join(folder_path, "fs0"), exist_ok=True)
        os.makedirs(os.path.join(folder_path, "fs1"), exist_ok=True)

        for idx, file in enumerate(file_list):
            old_path = os.path.join(folder_path, file)
            if idx % 2 == 0:
                new_path = os.path.join(folder_path, "fs0", "%06d.dcm" % (idx // 2))
            else:
                new_path = os.path.join(folder_path, "fs1", "%06d.dcm" % (idx // 2))
            move(old_path, new_path)

    # rebinning
    for dosage in ["full-dose", "low-dose"]:
        proj_md_path = os.path.join(opt.acr_dir, dosage + "-proj-params.pkl")
        proj_md = loadProjMetadata(proj_md_path)

        data_dir = os.path.join(opt.acr_dir, dosage + "-projs")
        rebinACR(data_dir, proj_md, device=opt.device, desc=dosage)

        # delete folder
        rmtree(data_dir)


if __name__ == "__main__":
    main()
