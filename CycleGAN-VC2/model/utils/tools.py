"""
Here is many useful tools.

1. load_dat: read htk files
2. file_to_batch_htk_and_labels
3. batch_to_pps
4. t_batch_to_onehot(targets, tl)
"""

import torch
import torch.nn as nn
import numpy as np
from struct import unpack, pack
import copy
import random

#DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_dat_htk(filename, frame_stack=True, spec=None):
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
    fh.seek(12, 0)
    dat = np.fromfile(fh, dtype='float32')
    dat = dat.reshape(int(len(dat) / veclen), veclen)
    dat = dat.byteswap()
    fh.close()

    dat = dat[:, :40]
    newlen = int(dat.shape[0] / 3)
    #frame_stacking
    if frame_stack:
        dat = dat[:3 * newlen, :]
        dat = np.reshape(dat, (newlen, 3, 40))
        dat = np.reshape(dat, (newlen, 3 * 40)).astype(np.float32)

    # spec-augmentation
    if spec is not None:
        aug_F = spec['F']
        aug_T = spec['T']

        # removing freq. bin
        aug_f_width = np.random.randint(0, aug_F)
        aug_f_start = np.random.randint(0, dat.shape[0] - aug_F)
        dat[:, aug_f_start:(aug_f_start + aug_f_width)] = 0.0
        #np.set_printoptions(threshold=10000)
        # remove time bin
        if dat.shape[0] > aug_T:
            aug_T_width = np.random.randint(0, aug_T)
            aug_T_start = np.random.randint(0, dat.shape[0] - aug_T)
            dat[aug_T_start:(aug_T_start + aug_T_width), :] = 0.0

    return dat.tolist()

def load_dat(filename, frame_stack=True, spec=False):
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


    dat = dat[:, :40]
    newlen = int(dat.shape[0] / 3)
    #frame_stacking
    if frame_stack:
        dat = dat[:3 * newlen, :]
        dat = np.reshape(dat, (newlen, 3, 40))
        dat = np.reshape(dat, (newlen, 3 * 40)).astype(np.float32)

    # spec-augmentation
    if spec == True:
        aug_F = 15
        aug_T = 100

        # removing freq. bin
        aug_f_width = np.random.randint(0, aug_F)
        aug_f_start = np.random.randint(0, dat.shape[0] - aug_F)
        dat[:, aug_f_start:(aug_f_start + aug_f_width)] = 0.0
        #np.set_printoptions(threshold=10000)
        # remove time bin
        if dat.shape[0] > aug_T:
            aug_T_width = np.random.randint(0, aug_T)
            aug_T_start = np.random.randint(0, dat.shape[0] - aug_T)
            dat[aug_T_start:(aug_T_start + aug_T_width), :] = 0.0

    return dat.tolist()

def dir_to_list(dir):
    id_list = []
    with open(dir) as f:
        for l in f:
            w, id = l.strip().split(' ')
            id_list.append([w, id])
    return id_list

def trans_id(input, id_l):
    for w, id in id_l:
        if input == w:
            return id
            break

def make_pps_batches(line_list, bs, line_num, iteration, sp=None):
    #st = time.time()
    bx, bt = list_to_batches(line_list, bs, line_num, iteration)
    #print('check1: ' + str(time.time() - st))
    px, tensor_bt, xl, tl = batch_to_pps(bx, bt)
    #print('check2: ' + str(time.time() - st))

    return px, tensor_bt, xl, tl

def make_pps_batches_with_names(line_list, bs, line_num, iteration):
    #st = time.time()
    bx, bt, n = list_to_batches_with_names(line_list, bs, line_num, iteration)
    #print('check1: ' + str(time.time() - st))
    px, tensor_bt, xl, tl, n = batch_to_pps_with_names(bx, bt, n)
    #print('check2: ' + str(time.time() - st))

    return px, tensor_bt, xl, tl, n


def list_to_batches(line_list, bs, line_num, iteration, shuffle=False, spec=False):
    """
    Input:
            str (scr_dir): script directory
            int (batch_s): batch size
            int (iter):    'n^th batch'
    Output:
            list (batch_x): R^(BATCH_SIZE * x_len * FEATURE_SIZE)
            list (batch_t): N^(BATCH_SIZE * t_len)
    """

    # Read and split into lines
    x_size = len(line_list)  # == FILE_LINE_NUM

    # make batch_x and batch_t
    batch_x = []
    batch_t = []

    # If the num of file lines is the n times of minibat size,
    # it causes error.
    is_dividable = False
    if line_num % bs == 0:
        is_dividable = True
    if iteration < int(line_num / bs):
        for i in range(bs):
            line = bs * iteration + i
            batch_x.append(load_dat(line_list[line][0], spec=spec))
            batch_t.append(list(map(int, (line_list[line][1]).strip().split(' '))))
    else: # append rest x
        if is_dividable:
            pass
        else:
            for i in range(x_size % bs):
                st = time.time()
                start = x_size - (x_size % bs)
                batch_x.append(load_dat(line_list[start + i][0], spec=spec))
                #batch_x.append(load_dat(line_list[start + i][0], sp))
                batch_t.append(str_to_int_list(line_list[start + i][1].split()))
    if shuffle:
        # For Transformer
        new_batch_x = []
        new_batch_t = []
        x = [i for i in range(bs)]
        random.shuffle(x)
        for ind in x:
            new_batch_x.append(batch_x[ind])
            new_batch_t.append(batch_t[ind])
        batch_x = new_batch_x
        batch_t = new_batch_t
    return batch_x, batch_t


def padding_batch(lx, lt):
    x_lens = []
    t_lens = []
    for x in lx:
        x_lens.append(len(x))
    for t in lt:
        t_lens.append(len(t))
    bx = torch.zeros(len(lx), max(x_lens), len(lx[0][0]))
    for i in range(len(lx)):
        padded_x = padding(lx[i], max(x_lens), [0]*len(lx[0][0]))
        padded_x = torch.FloatTensor(padded_x)
        bx[i] = padded_x
    bt = torch.zeros((len(lt), max(t_lens)), dtype=torch.int64)
    for i in range(len(lt)):
        padded_t = padding(lt[i], max(t_lens), 0)
        padded_t = torch.LongTensor(padded_t)
        bt[i] = padded_t
    return bx, torch.IntTensor(x_lens), bt, torch.IntTensor(t_lens)


def make_onehot_target(lt, class_size):
    t_lens = []
    for t in lt:
        t_lens.append(len(t))
    onehots = torch.zeros(len(lt), max(t_lens), class_size)
    for i in range(len(lt)):
        for ind, j in enumerate(lt[i]):
            tmp = int_to_onehot(j, class_size)
            tmp = torch.FloatTensor(tmp)
            onehots[i][ind] = tmp
    return onehots


def shift_list(lt):
    input = []
    truth = []
    for i in range(len(lt)):
        input.append(lt[i][1:])
        truth.append(lt[i][:-1])
    return input, truth


def list_to_batches_with_names(line_list, bs, line_num, iteration, sp):
    batch_x, batch_t = list_to_batches(line_list, bs, line_num, iteration, sp)
    names = list_to_speaker(line_list, bs, line_num, iteration)

    return batch_x, batch_t, names


def htk_file_names(line_list, bs, line_num, iteration):
    """
    To return htk file names
    """
    l = []
    for i in range(bs):
        line = bs * iteration + i
        l.append(line_list[line][0])
    return l

def list_to_speaker(line_list, bs, line_num, iteration):
    """
    x_list -> onehot_speakers
    """
    speaker_list = []
    for i in range(bs):
        line = bs * iteration + i
        l = line_list[line][0]
        if 'ainu' in l:
            utt = ''.join(list(l.split('/')[-2])[-2:])
        elif 'wsj' in l:
            utt = l.strip().split(' ', 1)[0].split('/')[-1].split('c0')[0].split('o0')[0]
        elif 'jnas' in l:
            utt = l.strip().split('/')[-3]
        else:
            print('ERROR: NOT AYNU, JNAS, or WSJ')
            exit()
        speaker_list.append(utt)
    return speaker_list

def htk_to_speaker(htk):
    utt = ''
    if 'ainu' in htk:
        utt = ''.join(list(htk.split('/')[-2])[-2:])
    elif 'wsj' in htk:
        utt = htk.strip().split(' ', 1)[0].split('/')[-1].split('c0')[0].split('o0')[0]
    elif 'jnas' in htk:
        utt = htk.strip().split('/')[-3]
    else:
        print('ERROR: NOT AYNU, JNAS, or WSJ')
        exit()
    return utt

def speaker_to_list(l, sp_list):
    r_l = []
    for s in l:
        r_l.append(trans_id(s, sp_list))
    #print(r_l)
    return str_to_int_list(r_l)


def does_all_contain_str(s, l):
    b = True
    for e in l:
        b = b and (s in e)
        # for all e, s is included.
    return b

def does_all_not_contain_str(s, l):
    b = True
    for e in l:
        b = b and not(s in e)
        # for all e, s is not included.
    return b

def batch_to_pps(x_batch, t_batch):
    """
    'pps' means 'pack_padded_sequence'
    from list_batch to tensor_batch (or pps) by padding

    input:
        list: [x1, x2, .. , xb] (b = BATCH_SIZE)
        list: [t1, t2, .. , tb] (      ,,      )

    output:
        PackedSequence:  x_pps:     R^(longest_x_length * FEATURE_SIZE * BATCH_SIZE)
        IntTensor:       t_tensor:  N^(BATCH_SIZE * lengest_t_length)
        IntTensor:       x_lengths: N^(BATCH_SIZE)
        IntTensor:       t_lengths: N^(BATCH_SIZE)
        (FEATURE_SIZE ( = 120) is the size of a frame of lmfb)
    """
    #
    # ----- Processing x (= acoustic feature) -----
    #
    orig_x_lengths = []
    for x in x_batch:
        orig_x_lengths.append(len(x))
    #print(orig_x_lengths)
    arg_sort_lengths = np.argsort(orig_x_lengths)
    #print(arg_sort_lengths)
    # Revercing is required by PyTorch's pps.
    rev_arg_sort_lengths = arg_sort_lengths[::-1]
    x_lengths = []
    padded_x_batch = []
    x_pad = [0] * len(x_batch[0][0])
    # x[0][0] = 120; FEATURE_SIZE
    for i in rev_arg_sort_lengths:
        padded_x = padding(x_batch[i], max(orig_x_lengths), x_pad)
        padded_x_batch.append(padded_x)
        x_lengths.append(orig_x_lengths[i])
    x_tensor = torch.tensor(padded_x_batch, dtype=torch.float32, requires_grad=True).cuda()
    #print(x_tensor.size())
    #print(x_lengths)
    x_pps = nn.utils.rnn.pack_padded_sequence(x_tensor, x_lengths, batch_first=True)
    x_lengths = torch.IntTensor(x_lengths).cuda()
    #
    # ---------- Processing t (= label) ----------
    #
    t_lengths = []
    # the same rev_arg_sort_lengths made by x for the consistency
    for i in rev_arg_sort_lengths:
        t_lengths.append(len(t_batch[i]))
    padded_t_batch = []
    t_pad = 0
    for i in rev_arg_sort_lengths:
        padded_t_batch.append(padding(t_batch[i], max(t_lengths), t_pad))
    t_tensor = torch.tensor(padded_t_batch).cuda()
    t_lengths = torch.IntTensor(t_lengths).cuda()

    return x_pps, t_tensor, x_lengths, t_lengths

def batch_to_pps_with_names(x_batch, t_batch, names):
    """
    'pps' means 'pack_padded_sequence'
    from list_batch to tensor_batch (or pps) by padding

    input:
        list: [x1, x2, .. , xb] (b = BATCH_SIZE)
        list: [t1, t2, .. , tb] (      ,,      )
        list: [n1, n2, .. , nb]

    output:
        PackedSequence:  x_pps:     R^(longest_x_length * FEATURE_SIZE * BATCH_SIZE)
        IntTensor:       t_tensor:  N^(BATCH_SIZE * lengest_t_length)
        IntTensor:       x_lengths: N^(BATCH_SIZE)
        IntTensor:       t_lengths: N^(BATCH_SIZE)
        list:            new_names: list(str)
        (FEATURE_SIZE ( = 120) is the size of a frame of lmfb)
    """
    #
    # ----- Processing x (= acoustic feature) -----
    #
    orig_x_lengths = []
    for x in x_batch:
        orig_x_lengths.append(len(x))
    #print(orig_x_lengths)
    arg_sort_lengths = np.argsort(orig_x_lengths)
    #print(arg_sort_lengths)
    # Revercing is required by PyTorch's pps.
    rev_arg_sort_lengths = arg_sort_lengths[::-1]
    x_lengths = []
    padded_x_batch = []
    x_pad = [0] * len(x_batch[0][0])
    # x[0][0] = 120; FEATURE_SIZE
    for i in rev_arg_sort_lengths:
        padded_x = padding(x_batch[i], max(orig_x_lengths), x_pad)
        padded_x_batch.append(padded_x)
        x_lengths.append(orig_x_lengths[i])
    x_tensor = torch.tensor(padded_x_batch, dtype=torch.float32, requires_grad=True).cuda()
    #print(x_tensor.size())
    #print(x_lengths)
    x_pps = nn.utils.rnn.pack_padded_sequence(x_tensor, x_lengths, batch_first=True)
    x_lengths = torch.IntTensor(x_lengths).cuda()
    #
    # ---------- Processing t (= label) ----------
    #
    t_lengths = []
    # the same rev_arg_sort_lengths made by x for the consistency
    for i in rev_arg_sort_lengths:
        t_lengths.append(len(t_batch[i]))
    padded_t_batch = []
    t_pad = 0
    for i in rev_arg_sort_lengths:
        padded_t_batch.append(padding(t_batch[i], max(t_lengths), t_pad))
    t_tensor = torch.tensor(padded_t_batch).cuda()
    t_lengths = torch.IntTensor(t_lengths).cuda()

    new_names = []
    for i in rev_arg_sort_lengths:
        new_names.append(names[i])

    return x_pps, t_tensor, x_lengths, t_lengths, new_names

"""
# This cannot work properly. At the same lengths, names con be changed.
def batch_to_pps_with_names(x_batch, t_batch, names):
    x_pps, t_tensor, x_lengths, t_lengths = batch_to_pps(x_batch, t_batch)
    orig_x_lengths = []
    for x in x_batch:
        orig_x_lengths.append(len(x))
    arg_sort_lengths = np.argsort(orig_x_lengths)
    # Reversing is required by PyTorch's pps.
    rev_arg_sort_lengths = arg_sort_lengths[::-1]
    new_names = []
    for i in rev_arg_sort_lengths:
        new_names.append(names[i])

    return x_pps, t_tensor, x_lengths, t_lengths, new_names
"""


# tools used above
def str_to_int_list(l):
    int_l = []
    for i in l:
        int_l.append(int(i))
    return int_l

def int_to_str_list(l):
    str_l = []
    for i in l:
        str_l.append(str(i))
    return str_l

def padding(l, length, pad):
    """
    i.e.
    input: [1, 2, 3], len = 5, pad = 0
    output: [1, 2, 3, 0, 0]
    """
    for i in range(length - len(l)):
        l.append(pad)
    return l


def int_to_onehot(label_num, class_num):
    """
    int -> list
    WARNING! the label_num must start '0'
    if label_num = 3, class_num = 10,
    onehot = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    """
    onehot = [0] * class_num
    onehot[label_num] = 1
    return onehot


def int_to_LS(label_num, class_num, LS_lambda):
    """
    int -> list
    WARNING! the label_num must start '0'
    if label_num = 3, class_num = 10,
    onehot = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    """
    onehot = [(1 - LS_lambda) / class_num] * class_num
    onehot[label_num] = LS_lambda

    return onehot


import time

def t_batch_to_onehot(targets, tl, word_size):
    """
    int_list -> Tensor
    """
    onehot_targets = np.zeros((len(targets), tl.max(), word_size))
    for i in range(len(targets)):
        for j in range(len(targets[0])):
            #a = int(round(time.time()*1000))
            #print(a)
            if j < tl[i]:
                onehot_targets[i][j][targets[i][j]] = 1
            else:
                # padding
                pass
    onehot_targets = torch.FloatTensor(onehot_targets).cuda()
    return onehot_targets

def t_batch_to_LS(targets, tl, word_size, LS_lambda):
    """
    int_list -> Tensor
    """
    #print(LS_lambda)
    #print(word_size)
    #print(tl.max())
    #print(len(targets))
    onehot_targets = np.array([[[(1 - LS_lambda) / word_size] * word_size] *  int(tl.max())] * len(targets))
    for i in range(len(targets)):
        for j in range(len(targets[0])):
            #a = int(round(time.time()*1000))
            #print(a)
            if j < tl[i]:
                onehot_targets[i][j][targets[i][j]] = LS_lambda
            else:
                # padding
                pass
    onehot_targets = torch.FloatTensor(onehot_targets).cuda()
    return onehot_targets

def cross_entropy(pred_dist, true_dist):
    """
    p: L*W (W: Word Size), already softmaxed
    q: L*W, onehot vector or all zero (padding)
    """
    loss = - (torch.log(pred_dist)*true_dist).sum()
    return loss

def batch_softmax(b):
    t = torch.zeros(b.shape).cuda()
    for i in range(len(b)):
        max_x = b[i].max()
        t[i] = b[i] - max_x
        exp_bi = torch.exp(t[i])
        sum_exp_bi = exp_bi.sum()
        t[i] = exp_bi / sum_exp_bi
    return t

def KL(pred_dist, true_dist):
    """
    p: L*W (W: Word Size), already softmaxed
    q: L*W, onehot vector or all zero (padding)
    """
    loss = - (torch.log(pred_dist)*true_dist).sum()
    # true_dist[0] = <SOS>, so you must not include this at comparing.
    return loss

def MSE(x1, x2):
    return ((x1 - x2)**2).sum()

def Cosine(x1, x2):
    x1 = x1.detach().cpu().numpy()
    x2 = x2.detach().cpu().numpy()
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

def init_weight(m):
    """
    To initialize weights and biases.
    """
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.uniform_(-0.1, 0.1)
        if isinstance(m.bias, nn.parameter.Parameter):
            m.bias.data.fill_(0)

    if classname.find('LSTM') != -1:
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param.data)
            if 'bias' in name:
                param.data.fill_(0)

    if classname.find('Conv') != -1 and classname != 'Conv2d_In_GLU':
        nn.init.kaiming_normal_(m.weight.data)
        if isinstance(m.bias, nn.parameter.Parameter):
            m.bias.data.fill_(0)


def file_to_list(filename):
    l = []
    with open(filename) as f:
        for line in f:
            l.append(line.strip())
    return(l)


def batch_to_lengths(batch):
    lens = []
    for e in batch:
        lens.append(len(e))
    return torch.IntTensor(lens).cuda()


def print_dict(d):
    """
    To print dictionary type.
    If the key contains 'align', it is not be printed.
    (Because it is only an alignment.)
    """
    for k in d:
        if k.find('align') == -1:
            print(k + ':\t' + str(d[k]))
        else:
            print(d[k])
    print()

def contractor(l, BLANK_ID):
    """
    For decoding CTC
    """
    prev = BLANK_ID
    con_list = []
    for i in l:
        if prev != i and i != BLANK_ID:
            con_list.append(int(i))
            prev = i
    return con_list

def list_to_str(l):
    s = ''
    for i in l:
        s = s + str(i) + ' '
    return s

def batch_to_ave(bx, xl):
    """
    B*T*H > B*H
    """
    ave_bx = torch.zeros((bx.shape[0], bx.shape[2]))
    for i in range(bx.shape[0]):
        ave_bx[i] = bx[i].sum(dim=0)# / xl[i]
    return ave_bx.cuda()


def get_clones(module, n):
    """
    To implement 'N-layer modules' in Transformer.\n
    by deep-copying and making a list of modules \n
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def tensor_to_img(t, filename):
    t = t.cpu()
    t = t.detach().numpy().T
    fig, ax = plt.subplots()
    ax.imshow(t, origin='lower')
    plt.savefig(filename, bbox_inches='tight', dpi=640)

def calc_param_num(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    pp=0
    for p in list(model_parameters):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp
