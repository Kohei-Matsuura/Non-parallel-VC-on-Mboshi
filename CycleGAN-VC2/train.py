r"""
To train Cycle-GAN-VC2
"""
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
TARGET_DIR = sys.argv[2] # KM
SAVE_DIR = sys.argv[3]

print('Source Script: {}'.format(SOURCE_DIR))
print('Target Script: {}'.format(TARGET_DIR))
print('Save Dir.: {}'.format(SAVE_DIR))

# Expanding hparams.py
hd = hparams.hd
tools.print_dict(hd)

# Seeding
random.seed(hd['SEED'])
np.random.seed(hd['SEED'])
torch.manual_seed(hd['SEED'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Import CycleGAN model
model = CycleGAN()
model.apply(tools.init_weight)
model.train()

# Sending the model to GPU if possible
DEVICE = torch.device('cuda' if torch.cuda.is_available else 'cpu')
model.to(DEVICE)

# Setting optimizers
D_params = list(model.Ds.parameters()) + list(model.Dt.parameters())
G_params = list(model.Gst.parameters()) + list(model.Gts.parameters())
opt_D = torch.optim.Adam(D_params,
                          lr=hd['LEARNING_RATE_D'],
                          betas=(0.5, 0.999),
                          weight_decay=hd['WEIGHT_DECAY'])
opt_G = torch.optim.Adam(G_params,
                          lr=hd['LEARNING_RATE_G'],
                          betas=(0.5, 0.999),
                          weight_decay=hd['WEIGHT_DECAY'])

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
# Target
target_dirs = []
with open(TARGET_DIR) as f:
    for l in f:
        target_dirs.append(l.strip())
while len(target_dirs)%hd['BATCH_SIZE'] != 0:
    target_dirs = target_dirs[:-1]
target_dirs_num = len(target_dirs)
print('Target num.: {}'.format(target_dirs_num))

print('{:-^20}'.format(' Start training... '))
start_time = int(round(time.time()))
for s in range(0, hd['STEP_NUM']):
    # Shuffling target data once in 10000 times
    if (s + 1) % 10000 == 0:
        random.shuffle(target_dirs)

    # Deleting identity loss
    if (s + 1) == hd['IDT_STOP_STEP']:
        hd['Lambda_idt'] = 0.0

    # Modifying features into tensors
    source = np.zeros((hd['BATCH_SIZE'],
                       hd['WIDTH'],
                       hd['FEATURE_SIZE']),
                       dtype=np.float32)
    for i in range(hd['BATCH_SIZE']):
        index = (s * hd['BATCH_SIZE'] % source_dirs_num) + i
        source_data = ctools.load_dat(source_dirs[index])
        #print(source_dirs[index])
        source[i, :, :] = ctools.random_pick(source_data, hd['WIDTH'])
    target = np.zeros((hd['BATCH_SIZE'],
                       hd['WIDTH'],
                       hd['FEATURE_SIZE']),
                       dtype=np.float32)
    for i in range(hd['BATCH_SIZE']):
        index = (s * hd['BATCH_SIZE'] % target_dirs_num) + i
        target_data = ctools.load_dat(target_dirs[index])
        target[i, :, :] = ctools.random_pick(target_data, hd['WIDTH'])

    # Passing through the model
    source = torch.tensor(source).to(DEVICE)
    target = torch.tensor(target).to(DEVICE)

    # Modifying parameters
    d = model(source, target)
    ## Discriminators: trained to discriminate properly
    ### Adversarial loss
    L_adv_D = 0.0
    L_adv_D += torch.mean((d['Ds_s'] - 1) ** 2)
    L_adv_D += torch.mean((d['Dt_t'] - 1) ** 2)
    L_adv_D += torch.mean((d['Ds_fake_s'] - 0) ** 2)
    L_adv_D += torch.mean((d['Dt_fake_t'] - 0) ** 2)
    L_adv_D /= 4.0 # Normalize
    ### backward
    opt_D.zero_grad()
    L_adv_D.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), hd['CLIPPING'])
    opt_D.step()

    ## Generator: trained to decieve D
    d = model(source, target)
    ### Adversarial loss
    L_adv_G = 0.0
    L_adv_G += torch.mean((d['Ds_fake_s'] - 1) ** 2)
    L_adv_G += torch.mean((d['Dt_fake_t'] - 1) ** 2)
    L_adv_G /= 2.0 # Normalize
    ### Identity loss
    L_idt = 0.0
    L_idt += torch.mean(torch.abs(source - d['idt_s']))
    L_idt += torch.mean(torch.abs(target - d['idt_t']))
    ### Cycle-consistency loss
    L_cyc = 0.0
    L_cyc += torch.mean(torch.abs(source - d['cyc_s']))
    L_cyc += torch.mean(torch.abs(target - d['cyc_t']))
    ### backward
    L_total = 0.0
    L_total = L_adv_G + hd['Lambda_idt'] * L_idt + hd['Lambda_cyc'] * L_cyc
    opt_G.zero_grad()
    L_total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), hd['CLIPPING'])
    opt_G.step()

    # Status print
    print()
    print('Step: {}'.format(s + 1))
    print('{} seconds have passed.'.format(int(round(time.time())) - start_time))
    print('Adversarial Loss (G): {}'.format(round(L_adv_G.item(), 3)))
    print('Adversarial Loss (D): {}'.format(round(L_adv_D.item(), 3)))
    print('Identity Loss: {}'.format(round(L_idt.item(), 3)))
    print('Cycle Loss: {}'.format(round(L_cyc.item(), 3)))
    sys.stdout.flush()

    # Saving
    if (s + 1) % hd['SAVE_PERIOD'] == 0 and (s + 1) > 0:
        torch.save(model.state_dict(), SAVE_DIR + '/params/step{}.net'.format(s + 1))
        #torch.save(opt_D.state_dict(), SAVE_DIR + '/params/step{}.opt_D'.format(s))
        #torch.save(opt_G.state_dict(), SAVE_DIR + '/params/step{}.opt_G'.format(s))
