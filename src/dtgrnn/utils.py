import dgl
import scipy.sparse as sparse
import numpy as np
import torch.nn as nn
import torch
import pandas as pd

class NormalizationLayer(nn.Module):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    # Here we shall expect mean and std be scaler
    def normalize(self, x):
        return (x-self.mean)/self.std

    def denormalize(self, x):
        return x*self.std + self.mean


def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0

    return loss.mean()


def get_learning_rate(optimizer):
    for param in optimizer.param_groups:
        return param['lr']

def get_limit_speed(node_list, limit_speed_rate):
    speed = pd.read_csv('../tsdlink_avgspeed_20211221.csv')
    speed = speed.set_index(keys=['tsdlinkid'], inplace=False,drop=['tsdlinkid'])
    avgspeed = pd.DataFrame(speed, columns=['avgspeed'])
    avgspeed = avgspeed.to_dict('index')
    limitSpeed = dict()
    for id in node_list:
        limitSpeed[id] = avgspeed[int(id)]['avgspeed'] * limit_speed_rate

    speed_list = list()
    for key in limitSpeed.keys():
        speed_list.append(limitSpeed[key])

    return np.array(speed_list).reshape(1, len(node_list))