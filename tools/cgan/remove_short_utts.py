import sys
import numpy as np
from struct import unpack, pack
from tqdm import tqdm
text_dir = sys.argv[1]
threshold = int(sys.argv[2])

lines = []

with open(text_dir) as f:
    for l in f:
        lines.append(l.strip())

def get_htk_data(filename):
    """
    To read binary data in htk file.
    The htk file includes log mel-scale filter bank.

    Args:
        filename : file name to read htk file

    Returns:
        dat : 120 (means log mel-scale filter bank) x T (time frame)
    """

    fh = open(filename, "rb")
    spam = fh.read(12)
    nSamples, sampPeriod, sampSize, parmKind = unpack(">IIHH", spam)
    veclen = int(sampSize / 4)

    return unpack(">IIHH", spam)

htk_lab_len = []
for htk in tqdm(lines):
    #len = int(htk.split('_')[-1].split('.')[0]) - int(htk.split('_')[1])
    if 'htk' in htk:
        num_sample = get_htk_data(htk)[0]
    elif 'npy' in htk:
        num_sample = np.load(htk).shape[0]
    if num_sample >= threshold:
        print(htk)
