import time

import numpy as np
import visdom


class Visualizer:

    def __init__(self, env="main", port=8097, **kwargs):
        self.vis = visdom.Visdom(port=port, env=env, **kwargs)
        self.index = {}
        self.log_text = ""

    def reinit(self, env="main", port=8097, **kwargs):
        self.vis = visdom.Visdom(port=port, env=env, **kwargs)

        return self

    # show a new log
    def log(self, info, win="log_text"):
        self.log_text += "[%s] %s <br>" % (time.strftime("%m%d_%H%M%S"), info)
        self.vis.text(self.log_text, win=win)

    # plot single data
    def plot(self, name, y):
        x = self.index.get(name, 0)
        self.vis.line(
            Y=np.array([y]), X=np.array([x]), win=name, opts=dict(title=name), update=None if x == 0 else "append"
        )
        self.index[name] = x + 1

    # plot multiple data
    def multiPlot(self, d):
        for k, v in d.items():
            self.plot(k, v)

    # plot single image
    def img(self, name, img):
        if len(img.size()) < 3:
            img = img.cpu().unsqueeze(0)
        self.vis.image(img.cpu(), win=name, opts=dict(title=name))

    # plot multiple images
    def multiImg(self, d):
        for k, v in d.items():
            self.img(k, v)

    # plot multiple image grids
    def img_grid_many(self, d):
        for k, v in d.items():
            self.img_grid(k, v)

    # other visdom methods
    def __getattr__(self, name):

        return getattr(self.vis, name)
