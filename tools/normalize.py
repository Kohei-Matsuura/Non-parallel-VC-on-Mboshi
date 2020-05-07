import sys
from load_dat import load_dat
import numpy as np
from tqdm import tqdm

FILE_LIST = sys.argv[1]
SAVE_DIR = sys.argv[2]
FEATURE_SIZE = int(sys.argv[3])

frame_num = 0
feat_sum = np.array([0] * FEATURE_SIZE, dtype='float')
feat_sqr_sum = np.array([0] * FEATURE_SIZE, dtype='float')

print('Loading data ...')
with open(FILE_LIST) as f:
    for l in tqdm(f):
        l = l.strip()
        filename = l.split('/')[-1]
        data = load_dat(l) # (T, 120)
        data = data[:, :FEATURE_SIZE]
        frame_num += data.shape[0]
        feat_sum += np.sum(data, axis=0)
        feat_sqr_sum += np.sum(data**2, axis=0)

feat_mean = feat_sum / frame_num
feat_mean_sqr = feat_mean ** 2
feat_sqr_mean = feat_sqr_sum / frame_num
feat_var = feat_sqr_mean - feat_mean_sqr

print('Normalizing data ...')
with open(FILE_LIST) as f:
    for l in tqdm(f):
        l = l.strip()
        filename = l.split('/')[-1]
        data = load_dat(l) # (T, 120)
        data = data[:, :FEATURE_SIZE]
        norm_data = (data - feat_mean) / np.sqrt(feat_var)
        save_dir = '{}/{}'.format(SAVE_DIR, filename)
        save_dir = '.'.join(save_dir.split('.')[:-1]) + '.npy'
        np.save(save_dir, norm_data)
