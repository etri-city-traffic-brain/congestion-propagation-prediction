import pickle
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score

class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True, shuffle=False):
        """

        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        if shuffle:
            permutation = np.random.permutation(self.size)
            xs, ys = xs[permutation], ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean
    
def getTimestamp(data):
    num_samples, num_nodes = data.shape
    time_ind = (data.index.values - data.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
    time_in_day = np.tile(time_ind, [num_nodes,1]).transpose((1, 0))
    return time_in_day

def getDayTimestamp(data):
    # 288 timeslots each day for tmap has 5 minutes time interval.
    df = pd.DataFrame({'timestamp':data.index.values})
    df['weekdaytime'] = df['timestamp'].dt.weekday * 288 + (df['timestamp'].dt.hour * 60 + df['timestamp'].dt.minute)//5
    df['weekdaytime'] = df['weekdaytime'] / df['weekdaytime'].max()
    num_samples, num_nodes = data.shape
    time_ind = df['weekdaytime'].values
    time_ind_node = np.tile(time_ind, [num_nodes,1]).transpose((1, 0))
    return time_ind_node

def getDayTimestamp_(start, end, freq, num_nodes):
    # 288 timeslots each day for tmap has 5 minutes time interval.
    df = pd.DataFrame({'timestamp':pd.date_range(start=start, end=end, freq=freq)})
    df['weekdaytime'] = df['timestamp'].dt.weekday * 288 + (df['timestamp'].dt.hour * 60 + df['timestamp'].dt.minute)//5
    df['weekdaytime'] = df['weekdaytime'] / df['weekdaytime'].max()
    time_ind = df['weekdaytime'].values
    time_ind_node = np.tile(time_ind, [num_nodes, 1]).transpose((1, 0))
    return time_ind_node

def masked_mse(preds, labels, null_val=1e-3):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels > null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=1e-3):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=1e-3):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels > null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=1e-3):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels > null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

# DCRNN
def masked_mae_loss(y_pred, y_true, y_mask):
    # mask = (y_true != 0).float()
    mask = (y_mask == 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()

def masked_mape_loss(y_pred, y_true, y_mask):
    # mask = (y_true != 0).float()
    mask = (y_mask == 0).float()
    mask /= mask.mean()
    loss = torch.abs(torch.div(y_true - y_pred, y_true))
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()

def masked_rmse_loss(y_pred, y_true, y_mask):
    # mask = (y_true != 0).float()
    mask = (y_mask == 0).float()
    mask /= mask.mean()
    loss = torch.pow(y_true - y_pred, 2)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return torch.sqrt(loss.mean())

def masked_mse_loss(y_pred, y_true, y_mask):
    # mask = (y_true != 0).float()
    mask = (y_mask == 0).float()
    mask /= mask.mean()
    loss = torch.pow(y_true - y_pred, 2)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()

def tsdlink_accracy_f1score(y_pred, y_true, speed_limit, tsdlink_list):
    # print('evaluate Accuracy score ...')
    acc = []
    f1 = []
    # print(y_true.shape)     # y_true.shape : torch.tensor(1,64,424,1)
    # print(y_pred.shape)     # y_pred.shape : torch.tensor(1,64,424,1)

    y_pred = y_pred[0]
    y_true = y_true[0]

    for i in range(len(y_true)):
        tsdlink_real_dict = defaultdict(list)
        tsdlink_pred_dict = defaultdict(list)

        # print(y_true[i].shape)      # y_true[i].shape : torch.tensor(424,1)
        # print(speed_limit.shape)    # speed_limist.shape : (1,424)

        true_sub = y_true[i].cpu().numpy().transpose() - speed_limit
        pred_sub = y_pred[i].cpu().numpy().transpose() - speed_limit

        real_congest = true_sub.copy()
        real_congest[real_congest >= 0] = 0
        real_congest[real_congest < 0] = 1

        pred_congest = pred_sub.copy()
        pred_congest[pred_congest >= 0] = 0
        pred_congest[pred_congest < 0] = 1

        for j in range(real_congest.shape[1]):
            t_l = tsdlink_list[0][j]
            real_congest_value = real_congest[0][j]
            pred_congest_value = pred_congest[0][j]
            for l in t_l:
                tsdlink_real_dict[l].append(real_congest_value)
                tsdlink_pred_dict[l].append(pred_congest_value)

        tsdlink_real_congest_list = list()
        tsdlink_pred_congest_list = list()

        for key in tsdlink_real_dict.keys():
            real_list = tsdlink_real_dict[key]
            real_list_1_count = real_list.count(1)
            real_list_0_count = real_list.count(0)
            # if real_list_1_count >= real_list_0_count:
            #     tsdlink_real_congest_list.append(1)
            # else:
            #     tsdlink_real_congest_list.append(0)
            if real_list_1_count > 0:
                tsdlink_real_congest_list.append(1)
            else:
                tsdlink_real_congest_list.append(0)

            pred_list = tsdlink_pred_dict[key]
            pred_list_1_count = pred_list.count(1)
            pred_list_0_count = pred_list.count(0)
            # if pred_list_1_count >= pred_list_0_count:
            #     tsdlink_pred_congest_list.append(1)
            # else:
            #     tsdlink_pred_congest_list.append(0)
            if pred_list_1_count > 0:
                tsdlink_pred_congest_list.append(1)
            else:
                tsdlink_pred_congest_list.append(0)

        acc.append(accuracy_score(tsdlink_real_congest_list, tsdlink_pred_congest_list))
        f1.append(f1_score(tsdlink_real_congest_list, tsdlink_pred_congest_list))

    ACC = np.array(acc).mean()
    F1 = np.array(f1).mean()

    return ACC, F1


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def print_params(model):
    # print trainable params
    param_count = 0
    print('Trainable parameter list:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape, param.numel())
            param_count += param.numel()
    print(f'\n In total: {param_count} trainable parameters. \n')
    return


def load_graph_data(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    return sensor_ids, sensor_id_to_ind, adj_mx