import os
from pickle import load
from random import choice, randint

import torch as t
from torch.utils import data


def readSample(file_path, manufacturer):
    with open(file_path, "rb") as file:
        file_data = load(file)

    # recon params
    theta = file_data["rebin_theta"]
    zloc = file_data["rebin_zloc"]

    # raw data
    if manufacturer == "Siemens":
        # extract projection data
        qd_proj0 = t.tensor(file_data["rebin_ld_data0"].astype("float32"))
        qd_proj1 = t.tensor(file_data["rebin_ld_data1"].astype("float32"))
        fd_proj0 = t.tensor(file_data["rebin_fd_data0"].astype("float32"))
        fd_proj1 = t.tensor(file_data["rebin_fd_data1"].astype("float32"))

        # extract noise quanta profile
        qd_noise0 = t.tensor(file_data["rebin_ld_noise0"]).unsqueeze(1).expand_as(qd_proj0)
        qd_noise1 = t.tensor(file_data["rebin_ld_noise1"]).unsqueeze(1).expand_as(qd_proj1)
        fd_noise0 = t.tensor(file_data["rebin_fd_noise0"]).unsqueeze(1).expand_as(fd_proj0)
        fd_noise1 = t.tensor(file_data["rebin_fd_noise1"]).unsqueeze(1).expand_as(fd_proj0)

        # interleave
        c, h, w = qd_proj0.size()
        qd_proj = t.empty((c, 2 * h, w))
        qd_proj[:, 0::2, :] = qd_proj0
        qd_proj[:, 1::2, :] = qd_proj1
        qd_noise = t.empty((c, 2 * h, w))
        qd_noise[:, 0::2, :] = qd_noise0
        qd_noise[:, 1::2, :] = qd_noise1

        c, h, w = fd_proj0.size()
        fd_proj = t.empty((c, 2 * h, w))
        fd_proj[:, 0::2, :] = fd_proj0
        fd_proj[:, 1::2, :] = fd_proj1
        fd_noise = t.empty((c, 2 * h, w))
        fd_noise[:, 0::2, :] = fd_noise0
        fd_noise[:, 1::2, :] = fd_noise1

        return qd_proj, qd_noise, fd_proj, fd_noise, theta, zloc

    elif manufacturer == "GE":
        # extract projection data
        qd_proj = t.tensor(file_data["rebin_ld_data"].astype("float32"))
        fd_proj = t.tensor(file_data["rebin_fd_data"].astype("float32"))

        # extract noise quanta profile
        qd_noise = t.tensor(file_data["rebin_ld_noise"]).unsqueeze(1).expand_as(qd_proj)
        fd_noise = t.tensor(file_data["rebin_fd_noise"]).unsqueeze(1).expand_as(fd_proj)

        return qd_proj, qd_noise, fd_proj, fd_noise, theta, zloc

    elif manufacturer == "ACR":
        # extract projection data
        proj0 = t.tensor(file_data["rebin_data0"].astype("float32"))
        proj1 = t.tensor(file_data["rebin_data1"].astype("float32"))

        # extract noise quanta profile
        noise0 = t.tensor(file_data["rebin_noise0"]).unsqueeze(1).expand_as(proj0)
        noise1 = t.tensor(file_data["rebin_noise1"]).unsqueeze(1).expand_as(proj1)

        # interleave
        c, h, w = proj0.size()
        proj = t.empty((c, 2 * h, w))
        proj[:, 0::2, :] = proj0
        proj[:, 1::2, :] = proj1
        noise = t.empty((c, 2 * h, w))
        noise[:, 0::2, :] = noise0
        noise[:, 1::2, :] = noise1

        return proj, noise, theta, zloc

    else:
        raise NotImplementedError()


def calcNoisePrior(qd_proj, ld_noise, fd_noise):
    out_proj = qd_proj
    out_noise = t.sqrt((1.0 / ld_noise - 1.0 / fd_noise) * t.exp(qd_proj))

    return out_proj, out_noise


def adjDoseLevel(qd_projs, fd_projs, fd_noise, alpha):
    # calculate dose level
    adj_dose = 1 / ((1 - alpha) ** 2 + 4 * alpha**2)

    # calculate noise quanta
    ld_noise = adj_dose * fd_noise

    # generate synthetic noisy data
    ld_projs = fd_projs + alpha * (qd_projs - fd_projs)

    return ld_projs, ld_noise


# projection training dataset
class TrainProjSet(data.Dataset):

    def __init__(self, opt):
        self.data_dir = opt.proj_dir
        self.patch_size = opt.patch_size
        self.buffer_size = opt.buffer_size
        self.study_list = opt.train_list
        self.manufacturer = opt.manufacturer
        self.alpha = opt.alpha

        # study properties
        self.sample_list = []
        for study in self.study_list:
            study_dir = os.path.join(self.data_dir, study, "rebin-projs")
            temp_list = [item for item in os.listdir(study_dir) if "pkl" in item]
            temp_list.sort()

            for index, f_name in enumerate(temp_list):
                self.sample_list.append((index, study, f_name))

    def __getitem__(self, index):
        # make sure index is within a study
        while True:
            (
                min_idx,
                _,
                _,
            ) = self.sample_list[index]
            (
                max_idx,
                _,
                _,
            ) = self.sample_list[index + self.buffer_size - 1]
            if max_idx - min_idx == self.buffer_size - 1:
                break
            else:
                index = index - 1

        qd_noise_list = []
        qd_projs_list = []
        fd_noise_list = []
        fd_projs_list = []
        for r_idx in range(self.buffer_size):
            _, study, f_name = self.sample_list[index + r_idx]

            # load sample data
            file_path = os.path.join(self.data_dir, study, "rebin-projs", f_name)
            qd_proj, qd_noise, fd_proj, fd_noise, _, _ = readSample(file_path, self.manufacturer)

            # add to buffers
            qd_noise_list.append(qd_noise)
            qd_projs_list.append(qd_proj)
            fd_noise_list.append(fd_noise)
            fd_projs_list.append(fd_proj)

        # repack to tensors
        qd_noise = t.stack(qd_noise_list, dim=0)
        qd_projs = t.stack(qd_projs_list, dim=0)
        fd_noise = t.stack(fd_noise_list, dim=0)
        fd_projs = t.stack(fd_projs_list, dim=0)

        # random cropping
        crop_c = choice([0, 1])
        crop_h = randint(0, qd_noise.size(-2) - self.patch_size[0])
        crop_w = randint(0, qd_noise.size(-1) - self.patch_size[1])

        qd_noise = qd_noise[
            :, crop_c, crop_h : crop_h + self.patch_size[0], crop_w : crop_w + self.patch_size[1]
        ].unsqueeze(1)
        qd_projs = qd_projs[
            :, crop_c, crop_h : crop_h + self.patch_size[0], crop_w : crop_w + self.patch_size[1]
        ].unsqueeze(1)
        fd_noise = fd_noise[
            :, crop_c, crop_h : crop_h + self.patch_size[0], crop_w : crop_w + self.patch_size[1]
        ].unsqueeze(1)
        fd_projs = fd_projs[
            :, crop_c, crop_h : crop_h + self.patch_size[0], crop_w : crop_w + self.patch_size[1]
        ].unsqueeze(1)

        # dose synthesis
        rand_alpha = (self.alpha[1] - self.alpha[0]) * t.rand(1) + self.alpha[0]
        ld_projs, ld_noise = adjDoseLevel(qd_projs, fd_projs, fd_noise, rand_alpha)

        # estimate noise prior
        ld_projs, ld_noise = calcNoisePrior(ld_projs, ld_noise, fd_noise)

        return ld_noise, ld_projs, fd_projs

    def __len__(self):

        return len(self.sample_list[: -(self.buffer_size - 1)])


# projection validation dataset
class ValProjSet(data.Dataset):

    def __init__(self, opt):
        self.data_dir = opt.proj_dir
        self.patch_size = opt.patch_size
        self.buffer_size = opt.buffer_size
        self.study_list = opt.val_list
        self.manufacturer = opt.manufacturer

        # study properties
        self.sample_list = []
        for study in self.study_list:
            study_dir = os.path.join(self.data_dir, study, "rebin-projs")
            temp_list = [item for item in os.listdir(study_dir) if "pkl" in item]
            temp_list.sort()

            for index, f_name in enumerate(temp_list):
                self.sample_list.append((index, study, f_name))

    def __getitem__(self, index):
        index = int(index * self.buffer_size)

        # make sure index is within a study
        while True:
            (
                min_idx,
                _,
                _,
            ) = self.sample_list[index]
            (
                max_idx,
                _,
                _,
            ) = self.sample_list[index + self.buffer_size - 1]
            if max_idx - min_idx == self.buffer_size - 1:
                break
            else:
                index = index - 1

        qd_noise_list = []
        qd_projs_list = []
        fd_noise_list = []
        fd_projs_list = []
        for r_idx in range(self.buffer_size):
            _, study, f_name = self.sample_list[index + r_idx]

            # load sample data
            file_path = os.path.join(self.data_dir, study, "rebin-projs", f_name)
            qd_proj, qd_noise, fd_proj, fd_noise, _, _ = readSample(file_path, self.manufacturer)

            # add to buffers
            qd_noise_list.append(qd_noise)
            qd_projs_list.append(qd_proj)
            fd_noise_list.append(fd_noise)
            fd_projs_list.append(fd_proj)

        # repack to tensors
        qd_noise = t.stack(qd_noise_list, dim=0)
        qd_projs = t.stack(qd_projs_list, dim=0)
        fd_noise = t.stack(fd_noise_list, dim=0)
        fd_projs = t.stack(fd_projs_list, dim=0)

        # central cropping
        crop_h = round((qd_noise.size(-2) - self.patch_size[0]) / 2)
        crop_w = round((qd_noise.size(-1) - self.patch_size[1]) / 2)

        qd_noise = qd_noise[:, 0, crop_h : crop_h + self.patch_size[0], crop_w : crop_w + self.patch_size[1]].unsqueeze(
            1
        )
        qd_projs = qd_projs[:, 0, crop_h : crop_h + self.patch_size[0], crop_w : crop_w + self.patch_size[1]].unsqueeze(
            1
        )
        fd_noise = fd_noise[:, 0, crop_h : crop_h + self.patch_size[0], crop_w : crop_w + self.patch_size[1]].unsqueeze(
            1
        )
        fd_projs = fd_projs[:, 0, crop_h : crop_h + self.patch_size[0], crop_w : crop_w + self.patch_size[1]].unsqueeze(
            1
        )

        # estimate noise prior
        qd_projs, qd_noise = calcNoisePrior(qd_projs, qd_noise, fd_noise)

        return qd_noise, qd_projs, fd_projs

    def __len__(self):

        return len(self.sample_list) // self.buffer_size


# projection test dataset
class TestProjSet(data.Dataset):

    def __init__(self, data_dir):
        self.data_dir = data_dir

        # load metadata
        self.sample_list = []
        md_path = os.path.join(self.data_dir, "../", "rebin-params.pkl")
        with open(md_path, "rb") as file:
            file_params = load(file)

        # scanner vendor
        self.manufacturer = file_params["manufacturer"]

        if self.manufacturer == "Siemens":
            self.fs0_w = file_params["fs0_w"]
            self.fs1_w = file_params["fs1_w"]

        elif self.manufacturer == "GE":
            self.fs_w = file_params["fs_w"]

        else:
            raise NotImplementedError()

        # find all samples
        self.sample_list = [file for file in os.listdir(self.data_dir) if "pkl" in file]
        self.sample_list.sort()

    def __getitem__(self, index):
        f_name = self.sample_list[index]

        # read sample
        file_path = os.path.join(self.data_dir, f_name)
        qd_proj, qd_noise, fd_proj, fd_noise, theta, zloc = readSample(file_path, self.manufacturer)

        # estimate noise prior
        qd_proj, qd_noise = calcNoisePrior(qd_proj, qd_noise, fd_noise)

        # interpolation weight
        rebin_w = t.zeros((4, qd_proj.size(1), qd_proj.size(2)))
        if self.manufacturer == "Siemens":
            fs0_w = t.tensor(self.fs0_w).unsqueeze(1)
            fs1_w = t.tensor(self.fs1_w).unsqueeze(1)
            rebin_w[:, 0::2, :] = fs0_w
            rebin_w[:, 1::2, :] = fs1_w

        elif self.manufacturer == "GE":
            fs_w = t.tensor(self.fs_w).unsqueeze(1)
            rebin_w[:, :, :] = fs_w

        else:
            raise NotImplementedError()

        return qd_noise, qd_proj, fd_proj, rebin_w, theta, zloc

    def __len__(self):

        return len(self.sample_list)


class ACRProjSet(data.Dataset):

    def __init__(self, dataset_dir, desc=""):
        self.data_dir = os.path.join(dataset_dir, desc + "-rebin-projs")

        # load metadata
        self.sample_list = []
        md_path = os.path.join(dataset_dir, desc + "-rebin-params.pkl")
        with open(md_path, "rb") as file:
            file_params = load(file)

        # scanner vendor is set to Siemens
        self.fs0_w = file_params["fs0_w"]
        self.fs1_w = file_params["fs1_w"]

        # find all samples
        self.sample_list = [file for file in os.listdir(self.data_dir) if "pkl" in file]
        self.sample_list.sort()

    def __getitem__(self, index):
        f_name = self.sample_list[index]

        # read sample
        file_path = os.path.join(self.data_dir, f_name)
        proj, noise, theta, zloc = readSample(file_path, "ACR")

        # estimate noise prior
        if "full-dose" in self.data_dir:
            out_proj = proj
            out_noise = noise
        else:
            out_proj, out_noise = calcNoisePrior(proj, noise, noise / 0.25)

        # interpolation weight
        rebin_w = t.zeros((4, out_proj.size(1), out_proj.size(2)))
        fs0_w = t.tensor(self.fs0_w).unsqueeze(1)
        fs1_w = t.tensor(self.fs1_w).unsqueeze(1)
        rebin_w[:, 0::2, :] = fs0_w
        rebin_w[:, 1::2, :] = fs1_w

        return out_noise, out_proj, rebin_w, theta, zloc

    def __len__(self):

        return len(self.sample_list)


# image training dataset
class TrainImgSet(data.Dataset):

    def __init__(self, opt):
        self.data_dir = opt.img_dir
        self.patch_size = opt.patch_size
        self.n_frames = opt.n_frames
        self.study_list = opt.train_list

        # study properties
        self.sample_list = []
        for study in self.study_list:
            file_list = [
                item for item in os.listdir(os.path.join(self.data_dir, study, "recon-imgs")) if item.endswith(".pkl")
            ]
            file_list.sort()
            for index, f_name in enumerate(file_list):
                self.sample_list.append((index, study, f_name))

    def __getitem__(self, index):
        # make sure index is within a study
        while True:
            min_idx, _, _ = self.sample_list[index]
            max_idx, _, _ = self.sample_list[index + self.n_frames - 1]
            if max_idx - min_idx == self.n_frames - 1:
                break
            else:
                index = index - 1

        noise_list = []
        qd_list = []
        fd_list = []
        mask_list = []
        for r_idx in range(self.n_frames):
            _, study, f_name = self.sample_list[index + r_idx]

            # load file
            file_path = os.path.join(self.data_dir, study, "recon-imgs", f_name)
            with open(file_path, "rb") as file:
                file_data = load(file)

            noise_img = t.tensor(file_data["noise"])
            qd_img = t.tensor(file_data["ld"])
            fd_img = t.tensor(file_data["fd"])
            mask = t.tensor(file_data["mask"])

            # add to buffers
            noise_list.append(noise_img.unsqueeze(0))
            qd_list.append(qd_img.unsqueeze(0))
            fd_list.append(fd_img.unsqueeze(0))
            mask_list.append(mask.unsqueeze(0))

        qd_noise = t.stack(noise_list, dim=0)
        qd_imgs = t.stack(qd_list, dim=0)
        fd_imgs = t.stack(fd_list, dim=0)
        masks = t.stack(mask_list, dim=0)

        # random cropping
        _, _, h, w = fd_imgs.size()
        crop_h = randint(0, h - self.patch_size[1])
        crop_w = randint(0, w - self.patch_size[1])
        qd_noise = qd_noise[:, :, crop_h : crop_h + self.patch_size[0], crop_w : crop_w + self.patch_size[1]]
        qd_imgs = qd_imgs[:, :, crop_h : crop_h + self.patch_size[0], crop_w : crop_w + self.patch_size[1]]
        fd_imgs = fd_imgs[:, :, crop_h : crop_h + self.patch_size[0], crop_w : crop_w + self.patch_size[1]]
        masks = masks[:, :, crop_h : crop_h + self.patch_size[0], crop_w : crop_w + self.patch_size[1]]

        return qd_noise, qd_imgs, fd_imgs, masks

    def __len__(self):

        return len(self.sample_list[: -(self.n_frames - 1)])


# image validation dataset
class ValImgSet(data.Dataset):

    def __init__(self, opt):
        self.data_dir = opt.img_dir
        self.patch_size = opt.patch_size
        self.n_frames = opt.n_frames
        self.study_list = opt.val_list

        # study properties
        self.sample_list = []
        for study in self.study_list:
            file_list = [
                item for item in os.listdir(os.path.join(self.data_dir, study, "recon-imgs")) if item.endswith(".pkl")
            ]
            file_list.sort()
            for index, f_name in enumerate(file_list):
                self.sample_list.append((index, study, f_name))

    def __getitem__(self, index):
        # make sure index is within a study
        while True:
            min_idx, _, _ = self.sample_list[index]
            max_idx, _, _ = self.sample_list[index + self.n_frames - 1]
            if max_idx - min_idx == self.n_frames - 1:
                break
            else:
                index = index - 1

        noise_list = []
        qd_list = []
        fd_list = []
        mask_list = []
        for r_idx in range(self.n_frames):
            _, study, f_name = self.sample_list[index + r_idx]

            # load file
            file_path = os.path.join(self.data_dir, study, "recon-imgs", f_name)
            with open(file_path, "rb") as file:
                file_data = load(file)

            noise_img = t.tensor(file_data["noise"])
            qd_img = t.tensor(file_data["ld"])
            fd_img = t.tensor(file_data["fd"])
            mask = t.tensor(file_data["mask"])

            # add to buffers
            noise_list.append(noise_img.unsqueeze(0))
            qd_list.append(qd_img.unsqueeze(0))
            fd_list.append(fd_img.unsqueeze(0))
            mask_list.append(mask.unsqueeze(0))

        qd_noise = t.stack(noise_list, dim=0)
        qd_imgs = t.stack(qd_list, dim=0)
        fd_imgs = t.stack(fd_list, dim=0)
        masks = t.stack(mask_list, dim=0)

        return qd_noise, qd_imgs, fd_imgs, masks

    def __len__(self):

        return len(self.sample_list[: -(self.n_frames - 1)])


# image test dataset
class TestImgSet(data.Dataset):

    def __init__(self, data_dir):
        self.data_dir = data_dir

        # find all samples
        self.sample_list = [item for item in os.listdir(self.data_dir) if item.endswith(".pkl")]
        self.sample_list.sort()

    def __getitem__(self, index):
        f_name = self.sample_list[index]

        # load file
        file_path = os.path.join(self.data_dir, f_name)
        with open(file_path, "rb") as file:
            file_data = load(file)

        qd_noise = t.tensor(file_data["noise"])
        qd_img = t.tensor(file_data["ld"])
        fd_img = t.tensor(file_data["fd"])
        mask = t.tensor(file_data["mask"])

        return qd_noise, qd_img, fd_img, mask

    def __len__(self):
        return len(self.sample_list)


class ACRImgSet(data.Dataset):

    def __init__(self, data_dir):
        self.data_dir = data_dir

        # find all samples
        self.sample_list = [item for item in os.listdir(self.data_dir) if item.endswith(".pkl")]
        self.sample_list.sort()

    def __getitem__(self, index):
        f_name = self.sample_list[index]

        # load file
        file_path = os.path.join(self.data_dir, f_name)
        with open(file_path, "rb") as file:
            file_data = load(file)

        noise = t.tensor(file_data["noise"])
        img = t.tensor(file_data["img"])
        mask = t.tensor(file_data["mask"])

        return noise, img, mask

    def __len__(self):
        return len(self.sample_list)
