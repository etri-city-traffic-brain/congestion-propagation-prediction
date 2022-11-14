import os
import ssl
from six.moves import urllib
import torch
import numpy as np
import dgl
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import scipy.sparse as sp


def download_file(dataset):
    print("Start Downloading data: {}".format(dataset))
    url = "https://s3.us-west-2.amazonaws.com/dgl-data/dataset/{}".format(
        dataset)
    print("Start Downloading File....")
    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)
    with open("./data/{}".format(dataset), "wb") as handle:
        handle.write(data.read())


class SnapShotDataset(Dataset):
    def __init__(self, path, npz_file):
        if not os.path.exists(path+'/'+npz_file):
            if not os.path.exists(path):
                os.mkdir(path)
            # download_file(npz_file)
        zipfile = np.load(path+'/'+npz_file)
        self.x = zipfile['x']
        self.y = zipfile['y']

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.x[idx, ...], self.y[idx, ...]


def METR_LAGraphDataset():
    if not os.path.exists('data/graph_la.bin'):
        if not os.path.exists('data'):
            os.mkdir('data')
        download_file('graph_la.bin')
    g, _ = dgl.load_graphs('data/graph_la.bin')
    print("************************")
    print(g[0])
    print("************************")

    return g[0]


class METR_LATrainDataset(SnapShotDataset):
    def __init__(self):
        super(METR_LATrainDataset, self).__init__('data', 'metr_la_train.npz')
        self.mean = self.x[..., 0].mean()
        self.std = self.x[..., 0].std()


class METR_LATestDataset(SnapShotDataset):
    def __init__(self):
        super(METR_LATestDataset, self).__init__('data', 'metr_la_test.npz')


class METR_LAValidDataset(SnapShotDataset):
    def __init__(self):
        super(METR_LAValidDataset, self).__init__('data', 'metr_la_valid.npz')


def PEMS_BAYGraphDataset():
    if not os.path.exists('data/graph_bay.bin'):
        if not os.path.exists('data'):
            os.mkdir('data')
        download_file('graph_bay.bin')
    g, _ = dgl.load_graphs('data/graph_bay.bin')
    return g[0]


class PEMS_BAYTrainDataset(SnapShotDataset):
    def __init__(self):
        super(PEMS_BAYTrainDataset, self).__init__(
            'data', 'pems_bay_train.npz')
        self.mean = self.x[..., 0].mean()
        self.std = self.x[..., 0].std()


class PEMS_BAYTestDataset(SnapShotDataset):
    def __init__(self):
        super(PEMS_BAYTestDataset, self).__init__('data', 'pems_bay_test.npz')


class PEMS_BAYValidDataset(SnapShotDataset):
    def __init__(self):
        super(PEMS_BAYValidDataset, self).__init__(
            'data', 'pems_bay_valid.npz')



def SEOUL_GraphDataset():
    # if not os.path.exists('data/graph_bay.bin'):
    #     if not os.path.exists('data'):
    #         os.mkdir('data')
    #     download_file('graph_bay.bin')
    # g, _ = dgl.load_graphs('data/graph_bay.bin')
    adj_mx = pd.read_csv('./dataset/Adj(urban-core).csv', header=None).values

    sp_mx = sp.coo_matrix(adj_mx)

    g = dgl.from_scipy(sp_mx)

    g.edata['weight'] = torch.ones(g.num_edges(), dtype=torch.int32)
    return g


class SEOUL_TrainDataset(SnapShotDataset):
    def __init__(self):
        super(SEOUL_TrainDataset, self).__init__(
            './dataset', 'seoul_train.npz')
        self.mean = self.x[..., 0].mean()
        self.std = self.x[..., 0].std()


class SEOUL_TestDataset(SnapShotDataset):
    def __init__(self):
        super(SEOUL_TestDataset, self).__init__('./dataset', 'seoul_test.npz')


class SEOUL_ValidDataset(SnapShotDataset):
    def __init__(self):
        super(SEOUL_ValidDataset, self).__init__(
            './dataset', 'seoul_val.npz')

def TMAP_GraphDataset(loc, loc_EN):
    # if not os.path.exists('data/graph_bay.bin'):
    #     if not os.path.exists('data'):
    #         os.mkdir('data')
    #     download_file('graph_bay.bin')
    # g, _ = dgl.load_graphs('data/graph_bay.bin')
    adj_mx = pd.read_csv('../tmap/{}/Adj({}_un).csv'.format(loc,loc_EN), header=None).values

    sp_mx = sp.coo_matrix(adj_mx)

    g = dgl.from_scipy(sp_mx)

    g.edata['weight'] = torch.ones(g.num_edges(), dtype=torch.int32)
    return g


class TMAP_TrainDataset(SnapShotDataset):
    def __init__(self, loc):
        super(TMAP_TrainDataset, self).__init__(
            './dataset', '{}_train.npz'.format(loc))
        self.mean = self.x[..., 0].mean()
        self.std = self.x[..., 0].std()


class TMAP_TestDataset(SnapShotDataset):
    def __init__(self,loc):
        super(TMAP_TestDataset, self).__init__('./dataset', '{}_test.npz'.format(loc))


class TMAP_ValidDataset(SnapShotDataset):
    def __init__(self,loc):
        super(TMAP_ValidDataset, self).__init__(
            './dataset', '{}_val.npz'.format(loc))

def data_transform(data, n_his, n_pred, device):
    # produce data slices for training and testing
    n_route = data.shape[1]
    l = len(data)
    num = l - n_his - n_pred
    x = np.zeros([num, 1, n_his, n_route])
    y = np.zeros([num, n_route])

    cnt = 0
    for i in range(l - n_his - n_pred):
        head = i
        tail = i + n_his
        x[cnt, :, :, :] = data[head: tail].reshape(1, n_his, n_route)
        y[cnt] = data[tail + n_pred - 1]
        cnt += 1
    return torch.Tensor(x).to(device), torch.Tensor(y).to(device)
