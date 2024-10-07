import numpy as np
import torch as t
from skimage.metrics import structural_similarity


class MaskL1Loss(t.nn.Module):

    def __init__(self, wc=1023.5, ww=4095.0):
        super(MaskL1Loss, self).__init__()
        self.data_min = wc - ww / 2
        self.data_max = wc + ww / 2
        self.criterion = t.nn.L1Loss(reduction="none")

    def forward(self, source, target, mask):
        valid_mask = (target >= self.data_min) * (target <= self.data_max) * mask
        valid_mask = valid_mask.float()
        loss = self.criterion(source, target)
        loss = t.sum(valid_mask * loss) / t.clip(t.sum(valid_mask[:]), min=1e-8)

        return loss


def calcMSE(src_img, tgt_img, wc, ww, mask):
    # compute lower and upper bounds
    min_hu = round(wc - ww / 2)
    max_hu = round(wc + ww / 2)

    # remove NaNs
    nan_mask = np.isnan(src_img)
    src_img[nan_mask] = 0.0

    # get valid mask within the window
    valid_mask = (tgt_img >= min_hu) * (tgt_img <= max_hu) * mask * ~nan_mask
    valid_mask = valid_mask.astype("float32")

    # compute mse
    mse = np.sum(valid_mask * (src_img - tgt_img) ** 2) / np.sum(valid_mask)

    return mse


def calcSSIM(src_img, tgt_img, wc, ww, mask):
    # compute lower and upper bounds
    min_hu = round(wc - ww / 2)
    max_hu = round(wc + ww / 2)

    # remove NaNs
    nan_mask = np.isnan(src_img)
    src_img[nan_mask] = 0.0

    # get valid mask within the window
    valid_mask = (tgt_img >= min_hu) * (tgt_img <= max_hu) * mask * ~nan_mask
    valid_mask = valid_mask.astype("float32")

    # compute ssim
    _, ssim_map = structural_similarity(src_img, tgt_img, data_range=4095.0, full=True)
    ssim = np.sum(valid_mask * ssim_map) / np.sum(valid_mask)

    return ssim
