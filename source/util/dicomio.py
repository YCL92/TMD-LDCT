import os
from datetime import datetime
from pickle import load

import numpy as np
import pydicom as pydcm
import torch as t

from .datautil import calcAngle, calcWeight, findNearest


# extract image metadata
def getImgMeta(data_dir, file_list):
    meta_dict = {}

    # read file
    file = pydcm.dcmread(os.path.join(data_dir, file_list[0]))

    # UIDs
    meta_dict["StudyInstanceUID"] = file.StudyInstanceUID
    meta_dict["SeriesInstanceUID"] = file.SeriesInstanceUID

    # scanner attributes
    meta_dict["Manufacturer"] = file.Manufacturer
    meta_dict["KVP"] = file.KVP
    meta_dict["XRayTubeCurrent"] = file.XRayTubeCurrent
    meta_dict["ExposureTime"] = file.ExposureTime

    # slice attributes
    meta_dict["SliceThickness"] = file.SliceThickness
    meta_dict["DataCollectionDiameter"] = file.DataCollectionDiameter
    meta_dict["ReconstructionDiameter"] = file.ReconstructionDiameter

    # pixel data attributes
    meta_dict["Rows"] = file.Rows
    meta_dict["Columns"] = file.Columns
    meta_dict["PixelSpacing"] = file.PixelSpacing

    # display attributes
    meta_dict["WindowCenter"] = file.WindowCenter
    meta_dict["WindowWidth"] = file.WindowWidth

    # patient attributes
    meta_dict["PatientName"] = file.PatientName
    meta_dict["PatientID"] = file.PatientID
    meta_dict["PatientSex"] = file.PatientSex
    meta_dict["PatientAge"] = file.PatientAge

    # per sample specific params
    meta_dict["ImagePositions"] = []
    meta_dict["XRayTubeCurrent"] = []
    meta_dict["Exposure"] = []
    for file_name in file_list:
        file = pydcm.dcmread(os.path.join(data_dir, file_name))
        meta_dict["ImagePositions"].append(file.ImagePositionPatient)
        meta_dict["XRayTubeCurrent"].append(file.XRayTubeCurrent)
        meta_dict["Exposure"].append(file.Exposure)

    return meta_dict


# extract Siemens projection metadata
def getSiemensProjMeta(data_dir, file_list):
    # projection metadata
    file = pydcm.dcmread(os.path.join(data_dir, file_list[0]))
    fs0 = [
        np.frombuffer(file[0x7033, 0x100D].value, dtype="float32"),
        np.frombuffer(file[0x7033, 0x100B].value, dtype="float32"),
        np.frombuffer(file[0x7033, 0x100C].value, dtype="float32"),
    ]  # focal spot 0 coordinates
    file = pydcm.dcmread(os.path.join(data_dir, file_list[1]))
    fs1 = [
        np.frombuffer(file[0x7033, 0x100D].value, dtype="float32"),
        np.frombuffer(file[0x7033, 0x100B].value, dtype="float32"),
        np.frombuffer(file[0x7033, 0x100C].value, dtype="float32"),
    ]  # focal spot 1 coordinates

    r_f = np.frombuffer(file[0x7031, 0x1003].value, dtype="float32")[0]  # source to isocentre distance
    r_fd = np.frombuffer(file[0x7031, 0x1031].value, dtype="float32")[0]  # source to detector distance
    n_projs_2pi = np.frombuffer(file[0x7033, 0x1013].value, dtype="uint16")[0]  # projections per 2PI with ffs
    n_rows = np.frombuffer(file[0x7029, 0x1010].value, dtype="uint16")[0]  # number of detector rows
    n_chnls = np.frombuffer(file[0x7029, 0x1011].value, dtype="uint16")[0]  # number of detector chnls
    cen_eles = np.frombuffer(file[0x7031, 0x1033].value, dtype="float32")  # central elements

    # fixed params calculation
    kc = n_chnls - cen_eles[0]  # central channel
    kr = cen_eles[1]  # central row
    # kr = cen_eles[1] - 1  # central row
    delta_beta = 2 * np.arcsin(
        0.5 * np.frombuffer(file[0x7029, 0x1002].value, dtype="float32")[0] / r_fd
    )  # fan angle increment
    slice_width = r_f / r_fd * np.frombuffer(file[0x7029, 0x1006].value, dtype="float32")[0]  # collimated slice width
    z_rot = file[0x0018, 0x9311].value * n_rows * slice_width  # gantry travel per rotation
    ds = r_f / r_fd * np.frombuffer(file[0x7029, 0x1002].value, dtype="float32")[0]  # sample interval

    # calibration params
    slope = np.array(file[0x0028, 0x1053].value, dtype="float32").item()
    intercept = np.array(file[0x0028, 0x1052].value, dtype="float32").item()
    mu_water = np.array(file[0x7041, 0x1001].value, dtype="float32").item()

    # for calculating z_l
    rebin_q = (np.linspace(0, n_rows - 1, num=2 * n_rows) - kr).astype("float32")
    rebin_qtan = rebin_q * np.frombuffer(file[0x7029, 0x1006].value, dtype="float32")[0] / r_fd
    rebin_wq = calcWeight(rebin_q / ((n_rows - 1) / 2))

    # rebined params
    rebin_kc = (2 * n_chnls - 1) / 2  # rebined central channel
    rebin_pmax = rebin_kc * r_f * np.sin(delta_beta / 2).astype("float32")  # rebined max ray-to-isocenter distance

    # nominal betas
    beta_k0 = (np.arange(0, n_chnls) - kc) * delta_beta

    # focal spot 0 interpolation indices
    r_fr = r_f + fs0[0]
    delta_alpha = -r_fr * np.tan(fs0[1])
    beta_rk = np.arcsin((r_f / r_fr) * (np.arange(0, 2 * n_chnls) - rebin_kc) * np.sin(delta_beta / 2))
    beta_rk0 = calcAngle(
        -r_fr, -delta_alpha, -(r_fd * np.cos(beta_k0) + fs0[0]), -(r_fd * np.sin(beta_k0) + delta_alpha)
    )
    p_idx0 = findNearest(beta_rk0, beta_rk).astype("float32")
    dr_idx0 = (-(n_projs_2pi / (4 * np.pi)) * (fs0[1] + beta_rk)).astype("float32")

    # focal spot 1 interpolation indices
    r_fr = r_f + fs1[0]
    delta_alpha = -r_fr * np.tan(fs1[1])
    beta_rk = np.arcsin((r_f / r_fr) * (np.arange(0, 2 * n_chnls) - rebin_kc) * np.sin(delta_beta / 2))
    beta_rk0 = calcAngle(
        -r_fr, -delta_alpha, -(r_fd * np.cos(beta_k0) + fs1[0]), -(r_fd * np.sin(beta_k0) + delta_alpha)
    )
    p_idx1 = findNearest(beta_rk0, beta_rk).astype("float32")
    dr_idx1 = (-(n_projs_2pi / (4 * np.pi)) * (fs1[1] + beta_rk)).astype("float32")

    # rebinning params
    proj_offset = int(np.ceil(np.max(np.abs(np.concatenate([dr_idx0, dr_idx1])))))

    # save to dictionary
    meta_dict = {
        "manufacturer": "Siemens",
        "r_f": r_f,
        "n_rows": int(n_rows),
        "n_chnls": int(n_chnls),
        "n_projs_full": len(file_list),
        "n_projs_2pi": int(n_projs_2pi),
        "proj_offset": proj_offset,
        "slice_width": slice_width,
        "z_rot": z_rot.astype("float32"),
        "ds": ds,
        "slope": slope,
        "intercept": intercept,
        "mu_water": mu_water,
        "rebin_qtan": rebin_qtan.astype("float32"),
        "rebin_wq": rebin_wq.astype("float32"),
        "rebin_pmax": rebin_pmax.astype("float32"),
        "p_idx0": p_idx0.astype("float32"),
        "dr_idx0": dr_idx0.astype("float32"),
        "p_idx1": p_idx1.astype("float32"),
        "dr_idx1": dr_idx1.astype("float32"),
    }

    return meta_dict


# extract GE projection metadata
def getGEProjMeta(data_dir, file_list):
    # projection params
    file = pydcm.dcmread(os.path.join(data_dir, file_list[0]))

    r_f = np.frombuffer(file[0x7031, 0x1003].value, dtype="float32")[0]  # source to isocentre distance
    r_fd = np.frombuffer(file[0x7031, 0x1031].value, dtype="float32")[0]  # source to detector distance
    n_projs_2pi = np.frombuffer(file[0x7033, 0x1013].value, dtype="uint16")[0]  # projections per 2PI
    n_rows = np.frombuffer(file[0x7029, 0x1010].value, dtype="uint16")[0]  # number of detector rows
    n_chnls = np.frombuffer(file[0x7029, 0x1011].value, dtype="uint16")[0]  # number of detector chnls
    cen_eles = np.frombuffer(file[0x7031, 0x1033].value, dtype="float32")  # central elements

    # fixed params calculation
    kc = cen_eles[0] - 1  # central channel
    kr = cen_eles[1] - 1  # central row
    delta_beta = 2 * np.arcsin(
        0.5 * np.frombuffer(file[0x7029, 0x1002].value, dtype="float32")[0] / r_fd
    )  # fan angle increment
    slice_width = r_f / r_fd * np.frombuffer(file[0x7029, 0x1006].value, dtype="float32")[0]  # collimated slice width
    z_rot = file[0x0018, 0x9311].value * n_rows * slice_width  # gantry travel per rotation
    ds = r_f / r_fd * np.frombuffer(file[0x7029, 0x1002].value, dtype="float32")[0]  # sample interval

    # calibration params
    slope = np.array(file[0x0028, 0x1053].value, dtype="float32").item()
    intercept = np.array(file[0x0028, 0x1052].value, dtype="float32").item()
    mu_water = np.array(file[0x7041, 0x1001].value, dtype="float32").item()

    # for calculating z_l
    rebin_q = np.arange(0, n_rows) - kr
    rebin_qtan = rebin_q * np.frombuffer(file[0x7029, 0x1006].value, dtype="float32")[0] / r_fd
    rebin_wq = calcWeight(rebin_q / ((n_rows - 1) / 2))

    # rebined params
    rebin_kc = (2 * n_chnls - 1) / 2  # rebined central channel
    rebin_pmax = rebin_kc * r_f * np.sin(delta_beta / 2).astype("float32")  # rebined max ray-to-isocenter distance

    # interpolation indices
    beta_k0 = (np.arange(0, n_chnls) - kc) * delta_beta
    beta_rk = np.arcsin((np.arange(0, 2 * n_chnls) - rebin_kc) * np.sin(delta_beta / 2))
    p_idx = findNearest(beta_k0, beta_rk)
    dr_idx = -(n_projs_2pi / (2 * np.pi)) * beta_rk

    # rebinning params
    proj_offset = int(np.ceil(np.max(np.abs(dr_idx))))

    # save to dictionary
    meta_dict = {
        "manufacturer": "GE",
        "r_f": r_f,
        "n_rows": int(n_rows),
        "n_chnls": int(n_chnls),
        "n_projs_full": len(file_list),
        "n_projs_2pi": int(n_projs_2pi),
        "proj_offset": proj_offset,
        "slice_width": slice_width,
        "z_rot": z_rot.astype("float32"),
        "ds": ds,
        "slope": slope,
        "intercept": intercept,
        "mu_water": mu_water,
        "rebin_qtan": rebin_qtan.astype("float32"),
        "rebin_wq": rebin_wq.astype("float32"),
        "rebin_pmax": rebin_pmax.astype("float32"),
        "p_idx": p_idx.astype("float32"),
        "dr_idx": dr_idx.astype("float32"),
    }

    return meta_dict


# load projection metadata
def loadProjMetadata(file_path):
    with open(file_path, "rb") as file:
        meta_data = load(file)

    meta_dict = {}
    meta_dict["manufacturer"] = meta_data["manufacturer"]
    meta_dict["n_rows"] = meta_data["n_rows"]
    meta_dict["n_chnls"] = meta_data["n_chnls"]
    meta_dict["n_projs_full"] = meta_data["n_projs_full"]
    meta_dict["n_projs_2pi"] = meta_data["n_projs_2pi"]
    meta_dict["proj_offset"] = meta_data["proj_offset"]
    meta_dict["r_f"] = t.tensor(meta_data["r_f"], dtype=t.float)
    meta_dict["z_rot"] = t.tensor(meta_data["z_rot"], dtype=t.float)
    meta_dict["ds"] = t.tensor(meta_data["ds"], dtype=t.float)
    meta_dict["slope"] = t.tensor(meta_data["slope"], dtype=t.float)
    meta_dict["intercept"] = t.tensor(meta_data["intercept"], dtype=t.float)
    meta_dict["mu_water"] = t.tensor(meta_data["mu_water"], dtype=t.float)
    meta_dict["rebin_qtan"] = t.tensor(meta_data["rebin_qtan"], dtype=t.float)
    meta_dict["rebin_wq"] = t.tensor(meta_data["rebin_wq"], dtype=t.float)
    meta_dict["rebin_pmax"] = t.tensor(meta_data["rebin_pmax"], dtype=t.float)

    try:
        meta_dict["p_idx0"] = t.tensor(meta_data["p_idx0"], dtype=t.float)
        meta_dict["dr_idx0"] = t.tensor(meta_data["dr_idx0"], dtype=t.float)
        meta_dict["p_idx1"] = t.tensor(meta_data["p_idx1"], dtype=t.float)
        meta_dict["dr_idx1"] = t.tensor(meta_data["dr_idx1"], dtype=t.float)
    except:
        meta_dict["p_idx"] = t.tensor(meta_data["p_idx"], dtype=t.float)
        meta_dict["dr_idx"] = t.tensor(meta_data["dr_idx"], dtype=t.float)

    return meta_dict


# load image metadata
def loadImgMetadata(file_path):
    with open(file_path, "rb") as file:
        meta_dict = load(file)

        # reconstruction locations
        meta_dict["recon_list"] = [float(item[-1]) for item in meta_dict["ImagePositions"]]

    return meta_dict


# save image as dicom file
def save2Dicom(img, img_idx, img_md, save_path, desc=""):
    # header
    file_md = pydcm.dataset.FileMetaDataset()
    file_md.TransferSyntaxUID = pydcm.uid.ExplicitVRLittleEndian
    file_md.MediaStorageSOPClassUID = pydcm.uid.CTImageStorage
    file_md.MediaStorageSOPInstanceUID = pydcm.uid.generate_uid()

    # check validity and create instance
    pydcm.dataset.validate_file_meta(file_md)
    ds = pydcm.dataset.FileDataset(
        save_path, {}, file_meta=file_md, preamble=b"\0" * 128, is_little_endian=True, is_implicit_VR=False
    )

    # UIDs
    ds.SOPClassUID = file_md.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_md.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = img_md["StudyInstanceUID"]
    ds.SeriesInstanceUID = img_md["SeriesInstanceUID"]
    ds.FrameOfReferenceUID = pydcm.uid.generate_uid()
    ds.SeriesDescription = desc

    # patient attributes
    ds.PatientName = "Anonymous"
    ds.PatientID = "Anonymous"
    ds.PatientBirthDate = ""
    ds.PatientSex = img_md["PatientSex"]
    ds.PatientAge = img_md["PatientAge"]

    # slice attributes
    ds.SliceThickness = img_md["SliceThickness"]
    ds.KVP = ""
    ds.ReconstructionDiameter = img_md["ReconstructionDiameter"]
    ds.DataCollectionDiameter = img_md["DataCollectionDiameter"]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.ImagePositionPatient = img_md["ImagePositions"][img_idx]

    # pixel data attributes
    ds.SpecificCharacterSet = "ISO_IR 100"
    ds.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL", "CT_SOM5 SPI"]
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.NumberOfFrames = "1"
    ds.Rows = img.shape[0]
    ds.Columns = img.shape[1]
    ds.PixelSpacing = img_md["PixelSpacing"]
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 0

    # display attributes
    ds.WindowCenter = img_md["WindowCenter"]
    ds.WindowWidth = img_md["WindowWidth"]
    ds.WindowCenterWidthExplanation = ""
    ds.RescaleIntercept = "-1024"
    ds.RescaleSlope = "1"

    # time attributes
    dt = datetime.now()
    dateStr = dt.strftime("%Y%m%d")
    timeStr = dt.strftime("%H%M%S.%f")
    ds.InstanceCreationDate = dateStr
    ds.InstanceCreationTime = timeStr
    ds.StudyDate = dateStr
    ds.StudyTime = timeStr
    ds.ContentDate = dateStr
    ds.ContentTime = timeStr

    # indexing attributes
    ds.StudyID = ""
    ds.SeriesNumber = "1"
    ds.InstanceNumber = str(img_idx + 1)

    # others
    ds.AccessionNumber = ""
    ds.Modality = "CT"
    ds.ReferringPhysicianName = ""
    ds.PositionReferenceIndicator = ""

    # image data
    img_int = np.round(img + 1024).astype("uint16")
    ds.PixelData = img_int.tobytes()

    # write to file
    ds.save_as(save_path)
