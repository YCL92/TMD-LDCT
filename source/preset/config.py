from torch import device
from torch.cuda import is_available


class Config:

    def __init__(self, mode="Test", manufacturer="Siemens"):
        # path to projection data, image data, results, and checkpoints
        self.acr_dir = "/mnt/share/data/ACR_Phantom_Data"
        self.proj_dir = "/mnt/share/data/LDCT-and-Projection-data"
        self.img_dir = "/mnt/share/data/LDCT-Image-data"
        self.result_dir = "../result"
        self.checkpoint_dir = "../checkpoint"
        self.manufacturer = manufacturer

        # mpd-net parameters
        if mode == "MPD-Net":
            self.patch_size = (64, 64)  # projection patch size
            self.n_frames = 3  # frame sequence length
            self.buffer_size = 5  # buffer size
            self.alpha = [0.5, 1.2]  # 17% ~ 80% of full dose
            self.lr = 1e-4  # learning rate
            self.lr_decay = 0.1  # learning rate decay
            self.max_epoch = 50  # maximum epoch
            self.n_val_samples = 128  # number of samples for validation
            self.val_freq = 1000  # validation frequency (iterations)

        # mir-net parameters
        elif mode == "MIR-Net":
            self.patch_size = (128, 128)  # image patch size
            self.n_frames = 7  # frame sequence length
            self.lr = 1e-4  # initial learning rate
            self.lr_decay = 0.1  # learning rate decay
            self.max_epoch = 500  # maximum epoch index

        # testing parameters
        elif mode == "Test":
            self.n_frames_mpd = 3  # frame sequence length for MPD-Net
            self.n_frames_mir = 7  # frame sequence length for MIR-Net
            self.n_interp = 3  # slice interpolation intervals

        else:
            raise NotImplementedError("Unknown mode: %s" % mode)

        # studies used for training, validation, and testing
        if self.manufacturer == "Siemens":
            train_list = [
                "L004", "L006", "L019", "L033", "L057", "L064", "L071", "L072", "L081", "L107",
                "L110", "L114", "L116", "L125", "L131", "L134", "L150", "L160", "L170", "L175",
                "L178", "L179", "L193", "L203", "L210", "L212", "L220", "L221", "L232", "L237",
                "L248", "L273", "L299",
            ]  # exclude L049 (incomplete)

            val_list = ["L229", "L148", "L077"]  # replace L148 low-dose-projs/fs1/014149.dcm (collapsed metadata)

            test_list = [
                "L014", "L056", "L058", "L075", "L123", "L145", "L186", "L187", "L209", "L219",
                "L241", "L266", "L277",
            ]

        elif self.manufacturer == "GE":
            train_list = [
                "L012", "L024", "L027", "L030", "L036", "L044", "L045", "L048", "L079", "L082",
                "L094", "L111", "L113", "L121", "L127", "L129", "L133", "L136", "L138", "L143",
                "L147", "L154", "L163", "L166", "L171", "L172", "L181", "L183", "L185", "L188",
                "L196", "L216",
            ]
            # exclude L035 (unmatched) and L144 (incorrect noise quanta)

            val_list = ["L043", "L213", "L238"]

            test_list = [
                "L218", "L228", "L231", "L234", "L235", "L244", "L250", "L251", "L257", "L260",
                "L267", "L269", "L288",
            ]

        else:
            raise NotImplementedError("Unsupported manufacturer: %s" % self.manufacturer)

        # add prefix
        self.train_list = [self.manufacturer + "-" + item for item in train_list]
        self.val_list = [self.manufacturer + "-" + item for item in val_list]
        self.test_list = [self.manufacturer + "-" + item for item in test_list]

        # other shared parameters
        self.seed = 282050419277614211  # random seed
        self.batch_size = 16  # mini-batch size
        self.num_workers = 0  # parallel cores
        self.device = device("cuda" if is_available() else "cpu")  # use cuda if available
        self.recon_filter = "shepp-logan"  # ramp, shepp-logan, cosine, hamming, or hann
