import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Discriminator_l2(nn.Module):
    def __init(self):
        super(Discriminator_l2, self).__init__()
        self.conv = nn.Conv2d(1, 20, 3)
        self.dense = nn.Linear(2048, 10)
