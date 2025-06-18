# Cross-domain Denoising for Low-dose Multi-frame Spiral Computed Tomography

If you find it useful in your research, please consider citing the following paper:

> ```tex
> @ARTICLE{10538291,
>   author={Lu, Yucheng and Xu, Zhixin and Hyung Choi, Moon and Kim, Jimin and Jung, Seung-Won},
>   journal={IEEE Transactions on Medical Imaging}, 
>   title={Cross-Domain Denoising for Low-Dose Multi-Frame Spiral Computed Tomography}, 
>   year={2024},
>   volume={43},
>   number={11},
>   pages={3949-3963},
>   keywords={Image reconstruction;Noise reduction;Computed tomography;Noise;Image denoising;Optimization;Spirals;Deep learning;low-dose computed tomography;image and video denoising},
>   doi={10.1109/TMI.2024.3405024}}
> ```

### 

### Updates:

2024-08-13: We have verified the GE section and uploaded the pre-trained weights.

2024-10-07: We have verified the Siemens section and uploaded the pre-trained weights.

2025-06-17: We have verified the Phantom section and updated the scripts accordingly.

### 

### Introduction

The main contributions of this work are as follows:

- We propose a two-stage framework for LDCT denoising. The proposed method works across both the projection and image domains. It is specifically optimized for CT scanners with multi-slice helical geometry.

- We model each stage's physical properties of noise and artifacts based on the data acquisition process in the reconstruction pipeline. This design improves the denoising performance and gives end-users richer interpretation ability and transparency.

- We demonstrate through experiments on patient data that our method significantly outperforms existing works both quantitatively and qualitatively. An extensive analysis of phantom scans further supports that the proposed method has achieved state-of-the-art performance.



------

Follow the instructions step by step to train the model from scratch. Or you can skip steps 3-5 to perform evaluations on patient data or phantom directly using the pre-trained models. 



#### 1. Projection Dataset Preparation

Visit [The Cancer Imaging Archive](https://www.cancerimagingarchive.net/collection/ldct-and-projection-data/) and follow the instructions to download the scans. Make sure to select only the abdomen scans, and uncheck the following studies as they won't be used in this project:

- L035 (unmatched series)
- L049 (incomplete study)
- L144 (incorrect noise quanta in metadata)

Also, download the phantom data from [The 2016 AAPM Grand Challenge](https://aapm.app.box.com/s/eaw4jddb53keg1bptavvvd1sf4x3pe9h) if you want to evaluate the spectral properties.

You will get two files ending with `.tcia`, follow [this link](https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images) for more information regarding dataset downloading on various platforms.

Due to the collapsed metadata in `Siemens-L148/low-dose-projs/fs1/014149.dcm`, you should replace it with `Siemens-L148/full-dose-projs/fs1/014149.dcm` as an alternative, note that study L148 is used for validation only so it does not affect the training process.

After that, update the patient dataset directory `proj_dir`, the phantom dataset directory parameter `acr_dir`, and the reconstructed result directory `img_dir`  in `./preset/config.py`.



##### *(optional) The CT Reconstruction Pipeline

If you are interested in the conventional CT image reconstruction pipeline from scratch (partially adopted and modified from the [FreeCT project](https://github.com/FreeCT/FreeCT) originally in C++ and R), run the following:

```bash
python recon-ge.py # for GE studies w/o FFS
python recon-siemens.py # for Siemens studies w/ FFS
```

The reconstruction results will be saved as DICOM files under `./result`, they can be opened by DICOM viewers (tested on [MicroDicom](https://www.microdicom.com/)).



#### 2. Pre-processing and Rebinning

After the dataset preparation is completed, apply pre-processing to the whole dataset by running the following:

```bash
python prepare-dataset.py # for the main dataset
python prepare-phantom.py # for the phantom scan
```

Then run the rebinning:

```bash
python rebin-ge.py # for GE studies
python rebin-siemens.py # for Siemens studies
```

The above commands will generate tons of small files that occupy plenty of space, make sure you have at least **1.7TB** of free disk space.



#### 3. Training MPD-Net

To train MPD-Net, specify the scanner model (either `manufacturer = Siemens` or `GE` in `./preset/config.py`), and run:

```bash
python s1-train-mpdnet.py
```

On our computer, the training takes a few days to complete. We notice some bugs when setting `num_workers > 1` on another machine with a different hardware setup, you may modify as needed.



#### 4. Image Dataset Preparation

When the training of MPD-Net is completed, generate the reconstructed image dataset for MIR-Net by running the following:

```bash
python s2-prep-recon.py
```

Wait until the image-domain dataset is well-prepared. This will take hours depending on your hardware configuration, if you are running out of space at this step, you can safely delete the projection files used to train MPD-Net **AFTER** reconstruction.



#### 5. Training MIR-Net

Finally, train MIR-Net by executing the following:

```bash
python s3-train-mirnet.py
```

On our computer, the training takes about 12 hours to complete.



#### 6. Evaluation of Patient Data

When all the training is completed, run the line below to evaluate the test set (change the manufacturer in `./preset/config.py` if needed, only valid if you use our provided weights):

```bash
python s4-run-test.py
```

Wait until all the test data is processed, then run the line below to print quantitative results:

```bash
python eval-testset.py
```

Due to the update of the reconstruction code and the GPU differences you will observe slightly different values than those reported in the paper.



#### 7. Evaluation of Phantom Data

Download `ACR_Phantom_Data.zip` from [The 2016 AAPM Grand Challenge](https://aapm.app.box.com/s/eaw4jddb53keg1bptavvvd1sf4x3pe9h) and unzip to `acr_dir` in `./preset/config.py`. Then run the following to prepare the phantom data:

```bash
python prepare-phantom.py
```

After that, run the following to get the results:

```bash
python eval-phantom.py
```

The above script will print out the statistical evaluation results in terms of MSE and SSIM, and save the corresponding images in DICOM format for further analysis.



#### 8. Compute NPS and TTF

The CT phantom evaluation tool is available at [iQMetrix-CT](https://github.com/SFPM/iQMetrix-CT), the two configuration files for calculating NPS and TTF reported in the paper are located at:

```bash
./preset/ACR 464_NPS_ROIs_Position_20231010_DFOV_250_MatrixSize_512.json

./preset/ACR 464_TTF_Inserts_Position_20231012_DFOV_250_MatrixSize_512.json
```
