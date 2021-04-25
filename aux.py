import torch
import numpy as np
import json
import glob

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_

from data_iter import real_data_loader, dis_data_loader, dis_l2_data_loader, vector_data_loader
from utils import recurrent_func, loss_func, get_sample, get_rewards
from Discriminator import Discriminator, Discriminator_l2
from Generator import Generator, Generator_l2
from target_lstm import TargetLSTM
from main import prepare_model_dict, prepare_optimizer_dict, prepare_scheduler_dict, pretrain_discriminator_l2

def get_params(filePath):
    with open(filePath, 'r') as f:
        params = json.load(f)
    f.close()
    return params

def restore_checkpoint():
    checkpoint = torch.load("checkpoints/checkpoint.pth.tar")
    return checkpoint

ckpt = restore_checkpoint()
model_dict = ckpt["model_dict"]

dis_l2_data_params = get_params("params/dis_l2_data_params.json")
train_params = get_params("params/train_l2_params.json")

lr_dict = train_params["lr_dict"]
gamma = train_params["decay_rate"]
step_size = train_params["decay_step_size"]

model_dict_l2 = prepare_model_dict("./params/leak_gan_l2_params.json", "l2", use_cuda=True)
optimizer_dict_l2 = prepare_optimizer_dict(model_dict_l2, lr_dict)
scheduler_dict_l2 = prepare_scheduler_dict(optimizer_dict_l2, gamma=gamma, step_size=step_size)

model_dict_l2, optimizer_dict_l2, scheduler_dict_l2 = pretrain_discriminator_l2(model_dict_l2,
                                                                                optimizer_dict_l2,
                                                                                scheduler_dict_l2,
                                                                                dis_l2_data_params,
                                                                                use_cuda=True)
