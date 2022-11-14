import torch.utils.data as utils

from Utils.graph import *
from Utils.math_ import *


def PrepareDataset(speed_matrix, BATCH_SIZE=100, seq_len=10, pred_len=1, train_propotion=0.7, valid_propotion=0.1,
                   z_norm=False):
    """ Prepare training and testing datasets and dataloaders.

    Convert speed/volume/occupancy matrix to training and testing dataset.
    The vertical axis of speed_matrix is the time axis and the horizontal axis
    is the spatial axis.

    Args:
        speed_matrix: a Matrix containing spatial-temporal speed data for a network
        seq_len: length of input sequence
        pred_len: length of predicted sequence
    Returns:
        Training dataloader
        Testing dataloader
    """
    time_len = speed_matrix.shape[0]

    max_speed = speed_matrix.max().max()
    #     speed_matrix =  speed_matrix / max_speed

    speed_sequences, speed_labels = [], []
    for i in range(time_len - seq_len - pred_len):
        speed_sequences.append(speed_matrix.iloc[i:i + seq_len].values)
        speed_labels.append(speed_matrix.iloc[i + seq_len:i + seq_len + pred_len].values)
    speed_sequences, speed_labels = np.asarray(speed_sequences), np.asarray(speed_labels)

    # shuffle and split the dataset to training and testing datasets
    sample_size = speed_sequences.shape[0]
    # index = np.arange(sample_size, dtype=int)
    # np.random.shuffle(index)

    train_index = int(np.floor(sample_size * train_propotion))
    valid_index = int(np.floor(sample_size * (train_propotion + valid_propotion)))

    train_data, train_label = speed_sequences[:train_index], speed_labels[:train_index]
    valid_data, valid_label = speed_sequences[train_index:valid_index], speed_labels[train_index:valid_index]
    test_data, test_label = speed_sequences[valid_index:], speed_labels[valid_index:]

    train_data, train_label = torch.Tensor(train_data), torch.Tensor(train_label)
    valid_data, valid_label = torch.Tensor(valid_data), torch.Tensor(valid_label)
    test_data, test_label = torch.Tensor(test_data), torch.Tensor(test_label)

    if z_norm:
        train_data = z_score(train_data)
        valid_data = z_score(valid_data)
        test_data = z_score(test_data)

    train_dataset = utils.TensorDataset(train_data, train_label)
    valid_dataset = utils.TensorDataset(valid_data, valid_label)
    test_dataset = utils.TensorDataset(test_data, test_label)

    train_dataloader = utils.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    valid_dataloader = utils.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    test_dataloader = utils.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    # test_dataloader = utils.DataLoader(test_dataset, batch_size=330, shuffle=False, drop_last=True)

    #     return train_dataloader, valid_dataloader, test_dataloader
    return train_dataloader, valid_dataloader, test_dataloader, max_speed


def load_weight_matrix(num_adj=3, direc='urban1_adj', plain=True, speed_limit=False,
                       speedCat=False, speedChange=False, dist=False, dist_og=False, angle=False, loc='Front'):
    # directory = '../tmap/{}'.format(loc) + '/'
    directory =  direc + '/'

    # directory = '/home/201702085/MW-TGC-PyTorch/SeoulData/' + direc + '/'
    # directory = '/media/hdd2/back_up/all files backup/whatever-Net/dataset/' + direc + '/'
    W = []
    if plain:
        for i in range(num_adj):
            PATH = directory + 'pl_Adj({}_un).csv'.format(loc)
            # PATH = 'pl_Adj({}_un).csv'.format(loc)

            tmp_sl_in = weight_matrix(PATH, prob=False, scaling=False)
            tmp_sl_in = tmp_sl_in + np.identity(len(tmp_sl_in))
            tmp_sl_in = tmp_sl_in / np.max(tmp_sl_in)

            PATH = directory + 'pl_Adj({}_un).csv'.format(loc)
            tmp_sl_out = weight_matrix(PATH, prob=False, scaling=False)
            tmp_sl_out = tmp_sl_out + np.identity(len(tmp_sl_out))
            tmp_sl_out = tmp_sl_out / np.max(tmp_sl_out)

            tmpW = np.stack((tmp_sl_in, tmp_sl_out), axis=2)

            if len(W) <= i:
                W.append(tmpW)
            else:
                W[i] = np.concatenate((W[i], tmpW), axis=2)

    if speed_limit:
        for i in range(num_adj):
            PATH = directory + 'sl_Adj({}_un).csv'.format(loc)
            tmp_sl_in = weight_matrix(PATH, prob=False, scaling=False)
            tmp_sl_in = tmp_sl_in + np.identity(len(tmp_sl_in))
            tmp_sl_in = tmp_sl_in / np.max(tmp_sl_in)

            PATH = directory + 'sl_Adj({}_un).csv'.format(loc)
            tmp_sl_out = weight_matrix(PATH, prob=False, scaling=False)
            tmp_sl_out = tmp_sl_out + np.identity(len(tmp_sl_out))
            tmp_sl_out = tmp_sl_out / np.max(tmp_sl_out)

            tmpW = np.stack((tmp_sl_in, tmp_sl_out), axis=2)

            if len(W) <= i:
                W.append(tmpW)
            else:
                W[i] = np.concatenate((W[i], tmpW), axis=2)

            '''
            ------------------------
            '''


    if speedCat:
        for i in range(num_adj):
            # PATH = directory + 'inADJ' + str(i + 1) + '_sl_ord.csv'
            PATH = directory + 'slc_Adj({}_un).csv'.format(loc)

            tmp_sl_in = weight_matrix(PATH, prob=False, scaling=False)
            tmp_sl_in = tmp_sl_in + np.identity(len(tmp_sl_in))
            tmp_sl_in = tmp_sl_in / np.max(tmp_sl_in)
            PATH = directory + 'slc_Adj({}_un).csv'.format(loc)

            # PATH = directory + 'outADJ' + str(i + 1) + '_sl_ord.csv'
            tmp_sl_out = weight_matrix(PATH, prob=False, scaling=False)
            tmp_sl_out = tmp_sl_out + np.identity(len(tmp_sl_out))
            tmp_sl_out = tmp_sl_out / np.max(tmp_sl_out)

            tmpW = np.stack((tmp_sl_in, tmp_sl_out), axis=2)

            if len(W) <= i:
                W.append(tmpW)
            else:
                W[i] = np.concatenate((W[i], tmpW), axis=2)

    if speedChange:
        for i in range(num_adj):
            PATH = directory + 'slcha_Adj({}_un).csv'.format(loc)
            # PATH = directory + 'inADJ' + str(i + 1) + '_sl_change.csv'
            tmp_sl_in = weight_matrix(PATH, prob=False, scaling=False)
            tmp_sl_in = tmp_sl_in + np.identity(len(tmp_sl_in))
            tmp_sl_in = tmp_sl_in / np.max(tmp_sl_in)
            PATH = directory + 'slcha_Adj({}_un).csv'.format(loc)
            # PATH = directory + 'outADJ' + str(i + 1) + '_sl_change.csv'
            tmp_sl_out = weight_matrix(PATH, prob=False, scaling=False)
            tmp_sl_out = tmp_sl_out + np.identity(len(tmp_sl_out))
            tmp_sl_out = tmp_sl_out / np.max(tmp_sl_out)

            tmpW = np.stack((tmp_sl_in, tmp_sl_out), axis=2)

            if len(W) <= i:
                W.append(tmpW)
            else:
                W[i] = np.concatenate((W[i], tmpW), axis=2)

    if dist:
        for i in range(num_adj):
            PATH = directory + 'Adj({}_dist).csv'.format(loc)
            # PATH = directory + 'doan_adj_0.5.csv'
            tmp_sl_in = weight_matrix(PATH, prob=False, scaling=False)
            tmp_sl_in = tmp_sl_in + np.identity(len(tmp_sl_in))
            tmp_sl_in = tmp_sl_in / np.max(tmp_sl_in)
            PATH = directory + 'Adj({}_dist).csv'.format(loc)
            # PATH = directory + 'doan_adj_0.5.csv'
            tmp_sl_out = weight_matrix(PATH, prob=False, scaling=False)
            tmp_sl_out = tmp_sl_out + np.identity(len(tmp_sl_out))
            tmp_sl_out = tmp_sl_out / np.max(tmp_sl_out)

            tmpW = np.stack((tmp_sl_in, tmp_sl_out), axis=2)

            if len(W) <= i:
                W.append(tmpW)
            else:
                W[i] = np.concatenate((W[i], tmpW), axis=2)


    if dist_og:
        for i in range(num_adj):
            PATH = directory + 'inADJ' + str(i + 1) + '_dist_og.csv'
            tmp_Win = weight_matrix(PATH, prob=False, scaling=False)
            tmp_Win = tmp_Win + np.identity(len(tmp_Win))
            tmp_Win = tmp_Win / np.max(tmp_Win)

            PATH = directory + 'outADJ' + str(i + 1) + '_dist_og.csv'
            tmp_Wout = weight_matrix(PATH, prob=False, scaling=False)
            tmp_Wout = tmp_Wout + np.identity(len(tmp_Wout))
            tmp_Wout = tmp_Wout / np.max(tmp_Wout)

            tmpW = np.stack((tmp_Win, tmp_Wout), axis=2)

            if len(W) <= i:
                W.append(tmpW)
            else:
                W[i] = np.concatenate((W[i], tmpW), axis=2)

    if angle:
        for i in range(num_adj):
            PATH = directory + 'inADJ' + str(i + 1) + '_angle.csv'
            tmp_angle_in = weight_matrix(PATH, prob=False, scaling=False)
            tmp_angle_in = tmp_angle_in + np.identity(len(tmp_angle_in))
            tmp_angle_in = tmp_angle_in / np.max(tmp_angle_in)

            PATH = directory + 'outADJ' + str(i + 1) + '_angle.csv'
            tmp_angle_out = weight_matrix(PATH, prob=False, scaling=False)
            tmp_angle_out = tmp_angle_out + np.identity(len(tmp_angle_out))
            tmp_angle_out = tmp_angle_out / np.max(tmp_angle_out)

            tmpW = np.stack((tmp_angle_in, tmp_angle_out), axis=2)

            if len(W) <= i:
                W.append(tmpW)
            else:
                W[i] = np.concatenate((W[i], tmpW), axis=2)

    if W == []:
        print("At least one weight must be True")
        raise
    return W