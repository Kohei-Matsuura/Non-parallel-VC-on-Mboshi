"""
From features list, return converted features
"""
# Basics
import sys
import numpy as np

# Tools
import model.utils.tools as tools
import model.utils.CycleGANTools as ctools
from tqdm import tqdm

# About PyTorch
import torch
import torch.nn as nn

# Hyper-parameters
import hparams

# Model
from model.CycleGAN import CycleGAN

# Getting args
FEAT_DIR = sys.argv[1]
MODEL_DIR = sys.argv[2]
DIRECTION = sys.argv[3] # 'st' or 'ts'
SAVE_DIR = sys.argv[4]

print('Features list: {}'.format(FEAT_DIR))
print('Save Dir.: {}'.format(SAVE_DIR))

# Expanding hparams.py
hd = hparams.hd

# Import CycleGAN model
model = CycleGAN()
params = torch.load(MODEL_DIR)
model.load_state_dict(params)
model.eval()

# Sending the model to GPU if possible
DEVICE = torch.device('cuda' if torch.cuda.is_available else 'cpu')
model.to(DEVICE)

# Getting Features
print('Getting Features...')
name_sources = []
with open(FEAT_DIR) as f:
    for l in tqdm(f):
        name = l.strip().split('/')[-1].split('.')[0]
        name_sources.append([name, ctools.load_dat(l.strip())])

# Converting
for i, [n, s] in enumerate(name_sources):
    s = s[:len(s) - len(s)%4, :]
    s = torch.FloatTensor(s).unsqueeze(0).to(DEVICE)
    #tools.tensor_to_img(s.squeeze(0).detach(), 'tmp' + '/{}.orig.png'.format(i))
    if DIRECTION == 'st':
        if i == 0:
            print('Using G_st...')
        fake_t = model.convert_st(s)
    elif DIRECTION == 'ts':
        if i == 0:
            print('Using G_ts...')
        fake_t = model.convert_ts(s)
    np_t = fake_t.squeeze(0).detach().cpu().numpy()
    #tools.tensor_to_img(fake_t.squeeze(0).detach(), 'tmp' + '/{}.fake.png'.format(i))
    np.save('{}/{}'.format(SAVE_DIR, n), np_t)
