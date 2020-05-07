import numpy as np
from struct import unpack, pack

def load_dat(filename):
    """
    To read binary data in htk or npy file.
    The htk file includes log mel-scale filter bank.

    Args:
        filename : file name to read htk file

    Returns:
        dat : 120 (means log mel-scale filter bank) x T (time frame)
    """

    if filename.split('.')[-1] == 'htk':
        fh = open(filename, "rb")
        spam = fh.read(12)
        nSamples, sampPeriod, sampSize, parmKind = unpack(">IIHH", spam)
        veclen = int(sampSize / 4)
        fh.seek(12, 0)
        dat = np.fromfile(fh, dtype='float32')
        dat = dat.reshape(int(len(dat) / veclen), veclen)
        dat = dat.byteswap()
        fh.close()
    elif filename.split('.')[-1] == 'npy':
        dat = np.load(filename)

    return dat # (T, 120)
