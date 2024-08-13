import os
from pickle import dump, HIGHEST_PROTOCOL
from shutil import move, rmtree

from pydicom import dcmread
from tqdm import tqdm

from preset.config import Config
from util.dicomio import getImgMeta, getSiemensProjMeta, getGEProjMeta


def main():
    # load config
    opt = Config()
    study_list = [
        item
        for item in os.listdir(opt.proj_dir)
        if os.path.isdir(os.path.join(opt.proj_dir, item))
        if "Siemens" not in item and "GE" not in item
    ]
    study_list.sort()

    # save only projections
    for study in tqdm(study_list, total=len(study_list)):
        study_path = os.path.join(opt.proj_dir, study)

        # find all folders
        folder_list = [item for item in os.listdir(study_path) if os.path.isdir(os.path.join(study_path, item))]

        for folder in folder_list:
            folder_path = os.path.join(opt.proj_dir, study, folder)

            # find all series
            series_list = [item for item in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, item))]

            for series in series_list:
                series_path = os.path.join(opt.proj_dir, study, folder, series)

                if "projections" in series.lower():
                    move(series_path, study_path)

                    # rename series
                    src_path = os.path.join(opt.proj_dir, study, series)
                    if "full dose" in series.lower():
                        dst_path = os.path.join(opt.proj_dir, study, "full-dose-projs")
                    elif "low dose" in series.lower():
                        dst_path = os.path.join(opt.proj_dir, study, "low-dose-projs")
                    else:
                        raise NotImplementedError("Unknown series: %s" % series)
                    os.rename(src_path, dst_path)

                elif "full dose images" in series.lower():
                    # extract image metadata
                    file_list = [item for item in os.listdir(series_path) if item.endswith(".dcm")]
                    file_list.sort()
                    img_meta = getImgMeta(series_path, file_list)
                    save_path = os.path.join(opt.proj_dir, study, "img-params.pkl")
                    with open(save_path, "wb") as file:
                        dump(img_meta, file, protocol=HIGHEST_PROTOCOL)

            # delete folder
            rmtree(folder_path)

        # extract projection metadata and move files to new folders
        fd_path = os.path.join(opt.proj_dir, study, "full-dose-projs")
        ld_path = os.path.join(opt.proj_dir, study, "low-dose-projs")
        file_list = [item for item in os.listdir(fd_path) if item.endswith(".dcm")]
        file_list.sort()

        # load basic scanner attributes
        head_file = dcmread(os.path.join(fd_path, file_list[0]))
        manufacturer = head_file.Manufacturer

        if "Siemens" in manufacturer.capitalize():
            manufacturer = manufacturer.capitalize()

            # extract projection metadata
            proj_meta = getSiemensProjMeta(fd_path, file_list)
            save_path = os.path.join(opt.proj_dir, study, "proj-params.pkl")
            with open(save_path, "wb") as file:
                dump(proj_meta, file, protocol=HIGHEST_PROTOCOL)

            # rename and move to new folders
            os.makedirs(os.path.join(fd_path, "fs0"), exist_ok=True)
            os.makedirs(os.path.join(fd_path, "fs1"), exist_ok=True)
            os.makedirs(os.path.join(ld_path, "fs0"), exist_ok=True)
            os.makedirs(os.path.join(ld_path, "fs1"), exist_ok=True)

            for idx, file in enumerate(file_list):
                # for full dose projections
                old_path = os.path.join(fd_path, file)
                if idx % 2 == 0:
                    new_path = os.path.join(fd_path, "fs0", "%06d.dcm" % (idx // 2))
                else:
                    new_path = os.path.join(fd_path, "fs1", "%06d.dcm" % (idx // 2))
                move(old_path, new_path)

                # for low dose projections
                old_path = os.path.join(ld_path, file)
                if idx % 2 == 0:
                    new_path = os.path.join(ld_path, "fs0", "%06d.dcm" % (idx // 2))
                else:
                    new_path = os.path.join(ld_path, "fs1", "%06d.dcm" % (idx // 2))
                move(old_path, new_path)

        elif "GE" in manufacturer:
            # extract projection metadata
            proj_meta = getGEProjMeta(fd_path, file_list)
            save_path = os.path.join(opt.proj_dir, study, "proj-params.pkl")
            with open(save_path, "wb") as file:
                dump(proj_meta, file, protocol=HIGHEST_PROTOCOL)

            for idx, file in enumerate(file_list):
                # for full dose projections
                old_path = os.path.join(fd_path, file)
                new_path = os.path.join(fd_path, "%06d.dcm" % idx)
                move(old_path, new_path)

                # for low dose projections
                old_path = os.path.join(ld_path, file)
                new_path = os.path.join(ld_path, "%06d.dcm" % idx)
                move(old_path, new_path)
        else:
            raise NotImplementedError("%s is not supported." % manufacturer)

        # rename study
        os.rename(os.path.join(opt.proj_dir, study), os.path.join(opt.proj_dir, manufacturer + "-" + study))


if __name__ == "__main__":
    main()
