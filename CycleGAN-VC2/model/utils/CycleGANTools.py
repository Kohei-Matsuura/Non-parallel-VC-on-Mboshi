"""
Tools for cycle-GAN
"""
import sys
import numpy as np
import torch
import torch.nn.utils.rnn as rnn
from struct import unpack, pack
import random

def load_dat(filename, frame_stacking=False):
    """
    To read binary data in htk or npy file.
    The htk file includes log mel-scale filter bank.

    Args:
        filename : file name to read htk file

    Returns:
        dat : (means log mel-scale filter bank) x T (time frame)
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
    dat = dat[:, :40] # removing delta and delta-delta
    #frame_stacking
    if frame_stacking:
        newlen = int(dat.shape[0] / 3)
        dat = dat[:3 * newlen, :]
        dat = np.reshape(dat, (newlen, 3, 40))
        dat = np.reshape(dat, (newlen, 3 * 40)).astype(np.float32)
    return dat


def add_mergins(features, mergin_width):
    """
    np.array: (L, F) -> np.array: (L-2M, 2M+1, F) \n
    From features to features with mergins \n
    i.e. \n
    [f1, f2, f3, f4, f5] -> [[f1, f2, f3], [f2, f3, f4], [f3, f4, f5]] \n
    """
    M = mergin_width
    L = features.shape[0]
    F = features.shape[1]
    new_features = np.zeros((L - 2 * M, 2 * M + 1, F))
    for i in range(L - 2 * M):
        C = i + M  # 'C' means 'central'
        new_features[i, :, :] = features[C-M:C+M+1, :]
    return new_features


def script_to_filelist(script_dir):
    """
    The script is supposed to be as below. \n
    (MEMO: use script in 'evals', that is, no labels)
        filename1\n
        filename2\n
        ...\n
        filenameN\n
    """
    filelist = []
    with open(script_dir) as f:
        for l in f:
            filelist.append(l.strip())
    return filelist


def divide_by_episode(filelist):
    """
    To divide filelist by episodes \n
    The filelist must not be sorted. \n
    """
    divided_list = []
    now_episode = ''
    for l in filelist:
        episode = l.split('/')[-2]
        if episode == now_episode:
            divided_list[-1].append(l)
        else:
            divided_list.append([l])
            now_episode = episode
    return divided_list


def filelist_to_features(filelist, feature_dim=40):
    """
    Stacking features from all files
    """
    stacked_features = np.array([0]*feature_dim)
    for l in filelist:
        f = load_dat(l)
        stacked_features = np.vstack((stacked_features, f))
    return stacked_features[1:]


def shuffle_features(features):
    """
    np.array: (L-2M, F) -> np.array: (L-2M, F) \n
    To shuffle feature blocks
    """
    feature_num = features.shape[0]
    shuffled_features = np.zeros(features.shape)
    random_indice = np.random.permutation(feature_num)
    for i, ind in enumerate(random_indice):
        shuffled_features[i] = features[ind]
    return shuffled_features


def shuffle_features_gpu(features):
    """
    np.array: (L-2M, F) -> np.array: (L-2M, F) \n
    To shuffle feature blocks
    """
    feature_num = features.shape[0]
    shuffled_features = torch.zeros(features.shape).cuda()
    random_indice = np.random.permutation(feature_num)
    for i, ind in enumerate(random_indice):
        shuffled_features[i] = features[ind]
    return shuffled_features


def prepare_features(dir, M_W, F, mergin_option, shuffle=True):
    """
    'mergin_option' = 'episode' / 'full' / 'utterance' \n
    With this, you can register what to be regarded as an vector.
    """
    filelist = script_to_filelist(dir)
    if mergin_option == 'episode':
        list_by_episode = divide_by_episode(filelist)
        all_features = np.zeros((1, M_W * 2 + 1, F))
        for i, e in enumerate(list_by_episode):
            episode_num = len(list_by_episode)
            print('Loading {}/{} episode...'.format(i+1, episode_num))
            epi_features = filelist_to_features(e)
            epi_featureblocks = add_mergins(epi_features, M_W)
            all_features = np.append(all_features, epi_featureblocks, axis=0)
        all_features = all_features[1:]
    elif mergin_option == 'full':
        all_features = filelist_to_features(filelist)
        all_features = add_mergins(all_features, M_W)
    elif mergin_option == 'utterance':
        all_features = np.zeros((1, M_W * 2 + 1, F))
        for u in filelist:
            features = filelist_to_features([u])
            featureblocks = add_mergins(features, M_W)
            all_features = np.append(all_features, featureblocks, axis=0)
        all_features = all_features[1:]
    else:
        print('NO SUCH OPTION: {}'.format(mergin_option))
        sys.exit()
    if shuffle:
        all_features = shuffle_features(all_features)
    return all_features


def gradient_penalty(net, real, fake):
    B, W, F = real.shape
    alpha = torch.rand(real.shape[0], 1)
    alpha = alpha.expand(B, W * F).contiguous().view(B, W, F)
    alpha = alpha.cuda()
    interpolates = real * alpha + fake * (1 - alpha)
    interpolates.requires_grad_(True)
    D_interpolates = net(interpolates)
    gradients = torch.autograd.grad(outputs=D_interpolates,
                                    inputs=interpolates,
                                    grad_outputs=torch.ones(D_interpolates.size()).cuda(),
                                    create_graph=True,
                                    retain_graph=True,
                                    only_inputs=True)
    gradients = gradients[0].view(B, -1) # flattening
    gradient_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def random_pick(features, N):
    """
    Randomly pick up N feature blocks. \n
    """
    idx = np.arange(len(features))
    np.random.shuffle(idx)
    sampled_idxs = idx[:N]
    return features[sampled_idxs]

def padding_feats(x_batch, t_batch):
    """
    input:
        conv_x_list: list(tensor)
        label_list: list(list(int))
    output:
        px, bt, xl, tl
    """
    orig_x_lengths = []
    for x in x_batch:
        orig_x_lengths.append(len(x))
    arg_sort_lengths = np.argsort(orig_x_lengths)
    # Revercing is required by PyTorch's pps.
    rev_arg_sort_lengths = arg_sort_lengths[::-1]
    x_lengths = []
    sorted_x_batch = []
    x_pad = [0] * len(x_batch[0][0])
    for i in rev_arg_sort_lengths:
        sorted_x_batch.append(x_batch[i])
        x_lengths.append(orig_x_lengths[i])
    x_tensor = rnn.pad_sequence(sorted_x_batch, batch_first=True)
    x_pps = rnn.pack_padded_sequence(x_tensor, x_lengths, batch_first=True)
    x_lengths = torch.IntTensor(x_lengths).cuda()

    t_lengths = []
    # the same rev_arg_sort_lengths made by x for the consistency
    for i in rev_arg_sort_lengths:
        t_lengths.append(len(t_batch[i]))
    sorted_t_batch = []
    t_pad = 0
    for i in rev_arg_sort_lengths:
        sorted_t_batch.append(t_batch[i])
    t_tensor = rnn.pad_sequence(sorted_t_batch, batch_first=True)
    t_lengths = torch.IntTensor(t_lengths).cuda()

    return x_pps, t_tensor, x_lengths, t_lengths

def random_pick(x,W):
    r"""
    (T, F) -> (W, F) where T > W
    """
    T, F = x.shape
    start_edge = random.randint(0, T - W)
    return x[start_edge:(start_edge + W), :]
