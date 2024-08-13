import os
import sys
import time

import torch as t


class BasicModules(t.nn.Module):

    def __init__(self):
        super(BasicModules, self).__init__()

    def load(self, save_dir=None, model_name=None, index=-1):
        if save_dir is None:
            save_dir = "../checkpoint"
        if model_name is None:
            model_name = self.model_name

        save_list = [file for file in os.listdir(save_dir) if file.startswith(model_name)]
        if len(save_list) == 0:
            sys.exit("Checkpoints not found!")
        save_list.sort()

        file_path = os.path.join(save_dir, save_list[index])
        state_dict = t.load(file_path, map_location=next(self.parameters()).device)
        self.load_state_dict(state_dict)
        print("Checkpoint loaded: %s" % file_path)

    def loadPartialDict(self, file_path):
        pretrained_dict = t.load(file_path, file_path, map_location=next(self.parameters()).device)
        model_dict = self.state_dict()
        pretrained_dict = {key: value for key, value in pretrained_dict.items() if key in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        print("Partial checkpoint loaded: %s" % file_path)

    def save(self, save_dir=None):
        if save_dir is None:
            save_dir = "../checkpoint"

        prefix = os.path.join("../checkpoint/" + self.model_name + "_")
        file_path = time.strftime(prefix + "%m%d-%H%M%S.pth")
        t.save(self.state_dict(), file_path)
        print("Checkpoint saved: %s" % file_path)
