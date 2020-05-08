# Basics
import sys
import random
import numpy as np
import time

# Handmade Tools
import model.utils.tools as tools
import model.utils.CycleGANTools as ctools

# Hyper-parameters
import hparams

# About PyTorch
import torch
import torch.nn as nn

# Model
from model.CycleGAN import CycleGAN

SOURCE_DIR = sys.argv[1] # KS
PARAM_DIR = sys.argv[2]
SAVE_DIR = 'converted_imgs'

print('Source Script: {}'.format(SOURCE_DIR))
print('Save Dir.: {}'.format(SAVE_DIR))

# Expanding hparams.py
hd = hparams.hd
#tools.print_dict(hd)

# Seeding
random.seed(hd['SEED'])
np.random.seed(hd['SEED'])
torch.manual_seed(hd['SEED'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Import CycleGAN model
model = CycleGAN()
model.apply(tools.init_weight)

params = torch.load(PARAM_DIR)
model.load_state_dict(params)

model.eval()

# Sending the model to GPU if possible
DEVICE = torch.device('cuda' if torch.cuda.is_available else 'cpu')
model.to(DEVICE)

# Reading scripts
print('Loading features...')
# Source
source_dirs = []
with open(SOURCE_DIR) as f:
    for l in f:
        source_dirs.append(l.strip())
while len(source_dirs)%hd['BATCH_SIZE'] != 0:
    source_dirs = source_dirs[:-1]
source_dirs_num = len(source_dirs)
print('Source num.: {}'.format(source_dirs_num))


print('{:-^20}'.format(' Start training... '))
start_time = int(round(time.time()))
for dir in source_dirs:
    name = dir.split('/')[-1]
    tmp_source = ctools.load_dat(dir)
    tmp_source = tmp_source[:len(tmp_source) - len(tmp_source)%4, :]
    tmp_source = torch.tensor(tmp_source).unsqueeze(0).to(DEVICE)
    generated_target = model.convert(tmp_source).squeeze(0).detach()
    tools.tensor_to_img(generated_target, SAVE_DIR + '/{}.png'.format(name))
    tmp_source = torch.tensor(ctools.load_dat(dir)).detach()
    tools.tensor_to_img(tmp_source, SAVE_DIR + '/{}.orig.png'.format(name))

