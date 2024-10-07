import os
from pickle import load

import numpy as np
from tqdm import tqdm

from preset.config import Config
from util.util import calcMSE, calcSSIM

# eval params
method_list = ["baseline", "prediction"]
wc = 40.0
ww = 300.0

opt = Config(mode="Test")
study_list = opt.test_list
study_list.sort()

# per-sample analysis
mse_results = {method: [] for method in method_list}
ssim_results = {method: [] for method in method_list}
for study in tqdm(study_list, total=len(study_list)):
    for method in method_list:
        # find all files
        file_list = [item for item in os.listdir(os.path.join(opt.result_dir, "baseline", study)) if "pkl" in item]
        file_list.sort()

        for file_name in file_list:
            # baseline images
            file_path = os.path.join(opt.result_dir, "baseline", study, file_name)
            with open(file_path, "rb") as file:
                all_data = load(file)

            fd_data = all_data["fd"].squeeze()
            ld_data = all_data["ld"].squeeze()
            mask = all_data["mask"].squeeze()

            if "baseline" in method:
                pred_data = ld_data
            else:
                file_path = os.path.join(opt.result_dir, method, study, file_name)
                with open(file_path, "rb") as file:
                    pred_data = load(file)

            # compute error
            mse_results[method].append(calcMSE(pred_data, fd_data, wc, ww, mask))
            ssim_results[method].append(calcSSIM(pred_data, fd_data, wc, ww, mask))

# print out results
for method in method_list:
    mse_avg = np.mean(mse_results[method])
    mse_std = np.std(mse_results[method])

    ssim_avg = np.mean(ssim_results[method])
    ssim_std = np.std(ssim_results[method])

    print("%s & $%.2f\pm%.2f$ & $%.4f\pm%.4f$" % (method, mse_avg, mse_std, ssim_avg, ssim_std))
