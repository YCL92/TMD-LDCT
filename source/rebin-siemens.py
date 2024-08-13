import os
from shutil import rmtree

from preset.config import Config
from util.dicomio import loadProjMetadata
from util.reconutil import rebinFFS


def main():
    # load config
    opt = Config(manufacturer="Siemens")
    study_list = opt.train_list + opt.val_list
    study_list.sort()

    # remove processed studies
    for study in study_list:
        study_dir = os.path.join(opt.proj_dir, study, "rebin-projs")
        if os.path.exists(study_dir):
            study_list.remove(study)

    # process each study
    for study in study_list:
        # load metadata
        proj_md_path = os.path.join(opt.proj_dir, study, "proj-params.pkl")
        proj_md = loadProjMetadata(proj_md_path)

        data_dir = os.path.join(opt.proj_dir, study)
        save_dir = os.path.join(opt.proj_dir, study, "rebin-projs")

        # run rebinning
        rebinFFS(data_dir, save_dir, proj_md, device=opt.device, desc=study)

        # delete folders
        rmtree(os.path.join(data_dir, "low-dose-projs"))
        rmtree(os.path.join(data_dir, "full-dose-projs"))


if __name__ == "__main__":
    main()
