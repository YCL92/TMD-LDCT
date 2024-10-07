import torch as t

from .basicmodule import BasicModules


# adaptive mixup
class AdptMixUp(BasicModules):

    def __init__(self):
        super(AdptMixUp, self).__init__()

        self.theta = t.nn.Parameter(data=t.tensor(0.5), requires_grad=True)

    def forward(self, x1, x2):
        w = t.sigmoid(self.theta)
        out = w * x1 + (1 - w) * x2

        return out


# residual U-Net
class ResUnet(BasicModules):

    def __init__(self, in_chnls, out_chnls, n_groups, n_feats, g_feats):
        super(ResUnet, self).__init__()

        # pre-processing
        self.head = t.nn.Sequential(
            t.nn.Conv2d(in_chnls * n_groups, g_feats * n_groups, 3, padding=1, groups=n_groups),
            t.nn.ReLU(inplace=True),
            t.nn.Conv2d(g_feats * n_groups, n_feats, 3, padding=1),
            t.nn.ReLU(inplace=True),
            t.nn.Conv2d(n_feats, n_feats, 3, padding=1),
            t.nn.ReLU(inplace=True),
        )

        # encoding blocks
        self.encode1 = t.nn.Sequential(
            t.nn.Conv2d(n_feats, 2 * n_feats, 3, padding=1, stride=2),
            t.nn.ReLU(inplace=True),
            t.nn.Conv2d(2 * n_feats, 2 * n_feats, 3, padding=1),
            t.nn.ReLU(inplace=True),
            t.nn.Conv2d(2 * n_feats, 2 * n_feats, 3, padding=1),
            t.nn.ReLU(inplace=True),
        )

        self.encode2 = t.nn.Sequential(
            t.nn.Conv2d(2 * n_feats, 4 * n_feats, 3, padding=1, stride=2),
            t.nn.ReLU(inplace=True),
            t.nn.Conv2d(4 * n_feats, 4 * n_feats, 3, padding=1),
            t.nn.ReLU(inplace=True),
            t.nn.Conv2d(4 * n_feats, 4 * n_feats, 3, padding=1),
            t.nn.ReLU(inplace=True),
        )

        # decoding blocks
        self.decode2 = t.nn.Sequential(
            t.nn.Conv2d(4 * n_feats, 4 * n_feats, 3, padding=1),
            t.nn.ReLU(inplace=True),
            t.nn.Conv2d(4 * n_feats, 4 * n_feats, 3, padding=1),
            t.nn.ReLU(inplace=True),
            t.nn.Conv2d(4 * n_feats, 8 * n_feats, 3, padding=1),
            t.nn.PixelShuffle(2),
        )

        self.decode1 = t.nn.Sequential(
            t.nn.Conv2d(2 * n_feats, 2 * n_feats, 3, padding=1),
            t.nn.ReLU(inplace=True),
            t.nn.Conv2d(2 * n_feats, 2 * n_feats, 3, padding=1),
            t.nn.ReLU(inplace=True),
            t.nn.Conv2d(2 * n_feats, 4 * n_feats, 3, padding=1),
            t.nn.PixelShuffle(2),
        )

        # post-processing
        self.tail = t.nn.Sequential(
            t.nn.Conv2d(n_feats, n_feats, 3, padding=1),
            t.nn.ReLU(inplace=True),
            t.nn.Conv2d(n_feats, n_feats, 3, padding=1),
            t.nn.ReLU(inplace=True),
            t.nn.Conv2d(n_feats, out_chnls, 1),
        )

        # adaptive mixup layers
        self.mix_up1 = AdptMixUp()
        self.mix_up2 = AdptMixUp()

    def forward(self, in_imgs, in_noise):
        x = []
        n_frames = in_imgs.size(1)
        for idx in range(n_frames):
            x.append(in_imgs[:, idx, :, :, :])
            x.append(in_noise[:, idx, :, :, :])
        x = t.cat(x, dim=1).contiguous()

        # head features
        feats0 = self.head(x)

        # down-sampling
        feats1 = self.encode1(feats0)
        feats2 = self.encode2(feats1)

        # up-sampling
        feats2 = self.decode2(feats2)
        feats1 = self.decode1(self.mix_up2(feats1, feats2))

        # tail features
        out_noise = self.tail(self.mix_up1(feats0, feats1))

        return out_noise


# multi-frame projection denoising network
class MPDNet(BasicModules):

    def __init__(self, n_frames=3, n_feats=64, g_feats=16, manufacturer="Siemens"):
        super(MPDNet, self).__init__()
        assert manufacturer in ["Siemens", "GE"]
        self.model_name = "MPD-Net-" + manufacturer
        self.n_frames = n_frames

        # denoising blocks
        self.stage1 = ResUnet(2, 1, self.n_frames, n_feats, g_feats)
        self.stage2 = ResUnet(2, 1, self.n_frames, n_feats, g_feats)

        # initialization
        for _, m in enumerate(self.modules()):
            if isinstance(m, t.nn.Conv2d):
                t.nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, img, noise, t_idx=0):
        # initiate buffers
        if t_idx == 0:
            self.s1i_buff = []
            self.s1n_buff = []
            self.s2i_buff = []
            self.s2n_buff = []
            self.tmp_buff = []

        # update stage 1 buffer
        self.s1i_buff.append(img)
        self.s1n_buff.append(noise)
        if len(self.s1n_buff) < self.n_frames:

            return None, None
        else:
            self.s1i_buff = self.s1i_buff[-self.n_frames :]
            self.s1n_buff = self.s1n_buff[-self.n_frames :]

        # stage 1 denoising
        in_imgs = t.stack(self.s1i_buff, dim=1)
        in_noise = t.stack(self.s1n_buff, dim=1)
        self.s2i_buff.append(in_imgs[:, self.n_frames // 2, :, :, :])
        self.s2n_buff.append(self.stage1(in_imgs, in_noise))

        # update stage 2 buffer
        if len(self.s2n_buff) < self.n_frames:

            return None, None
        else:
            self.s2i_buff = self.s2i_buff[-self.n_frames :]
            self.s2n_buff = self.s2n_buff[-self.n_frames :]

        # stage 2 denoising
        in_imgs = t.stack(self.s2i_buff, dim=1)
        in_noise = t.stack(self.s2n_buff, dim=1)
        out_img = in_imgs[:, self.n_frames // 2, :, :, :].clone()
        out_noise = self.stage2(in_imgs, in_noise)

        return out_img, out_noise


# multi-frame image refinement network
class MIRNet(BasicModules):

    def __init__(self, n_frames=7, n_feats=96, manufacturer="Siemens"):
        super(MIRNet, self).__init__()
        self.model_name = "MIR-Net-" + manufacturer
        in_chnls = 2 * n_frames

        # pre-processing
        self.head = t.nn.Sequential(
            t.nn.Conv2d(in_chnls, n_feats, 3, padding=1),
            t.nn.ReLU(inplace=True),
            t.nn.Conv2d(n_feats, n_feats, 3, padding=1),
            t.nn.ReLU(inplace=True),
            t.nn.Conv2d(n_feats, n_feats, 3, padding=1),
            t.nn.ReLU(inplace=True),
        )

        # encoding blocks
        self.encode1 = t.nn.Sequential(
            t.nn.Conv2d(n_feats, 2 * n_feats, 3, padding=1, stride=2),
            t.nn.ReLU(inplace=True),
            t.nn.Conv2d(2 * n_feats, 2 * n_feats, 3, padding=1),
            t.nn.ReLU(inplace=True),
            t.nn.Conv2d(2 * n_feats, 2 * n_feats, 3, padding=1),
            t.nn.ReLU(inplace=True),
        )

        self.encode2 = t.nn.Sequential(
            t.nn.Conv2d(2 * n_feats, 4 * n_feats, 3, padding=1, stride=2),
            t.nn.ReLU(inplace=True),
            t.nn.Conv2d(4 * n_feats, 4 * n_feats, 3, padding=1),
            t.nn.ReLU(inplace=True),
            t.nn.Conv2d(4 * n_feats, 4 * n_feats, 3, padding=1),
            t.nn.ReLU(inplace=True),
        )

        # decoding blocks
        self.decode2 = t.nn.Sequential(
            t.nn.Conv2d(4 * n_feats, 4 * n_feats, 3, padding=1),
            t.nn.ReLU(inplace=True),
            t.nn.Conv2d(4 * n_feats, 4 * n_feats, 3, padding=1),
            t.nn.ReLU(inplace=True),
            t.nn.Conv2d(4 * n_feats, 8 * n_feats, 3, padding=1),
            t.nn.PixelShuffle(2),
        )

        self.decode1 = t.nn.Sequential(
            t.nn.Conv2d(2 * n_feats, 2 * n_feats, 3, padding=1),
            t.nn.ReLU(inplace=True),
            t.nn.Conv2d(2 * n_feats, 2 * n_feats, 3, padding=1),
            t.nn.ReLU(inplace=True),
            t.nn.Conv2d(2 * n_feats, 4 * n_feats, 3, padding=1),
            t.nn.PixelShuffle(2),
        )

        # post-processing
        self.tail = t.nn.Sequential(
            t.nn.Conv2d(n_feats, n_feats, 3, padding=1),
            t.nn.ReLU(inplace=True),
            t.nn.Conv2d(n_feats, n_feats, 3, padding=1),
            t.nn.ReLU(inplace=True),
            t.nn.Conv2d(n_feats, 1, 1),
        )

        # adaptive mixup layers
        self.mix_up1 = AdptMixUp()
        self.mix_up2 = AdptMixUp()

    def forward(self, in_imgs, in_noise):
        x = []
        n_frames = in_imgs.size(1)
        for idx in range(n_frames):
            x.append(in_imgs[:, idx, :, :, :])
            x.append(in_noise[:, idx, :, :, :])
        x = t.cat(x, dim=1)

        # head features
        feats0 = self.head(x)

        # down-sampling
        feats1 = self.encode1(feats0)
        feats2 = self.encode2(feats1)

        # up-sampling
        feats2 = self.decode2(feats2)
        feats1 = self.decode1(self.mix_up2(feats1, feats2))

        # tail features
        out_noise = self.tail(self.mix_up1(feats0, feats1))
        out_img = in_imgs[:, n_frames // 2, :, :, :].clone()

        return out_img, out_noise
