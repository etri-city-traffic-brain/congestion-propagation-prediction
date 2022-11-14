import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import sys


class Encoder_LSTM(nn.Module):
    """
    An implementation of (MW-TGC) + (Encoder LSTM Cell).
    """

    def __init__(self, input_size, hidden_size, out_features, out_channels, adj, GCN=False, bias=False):
        super(Encoder_LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)

        self.bn_xh = nn.BatchNorm1d(4 * self.hidden_size, affine=False)
        self.bn_hh = nn.BatchNorm1d(4 * self.hidden_size, affine=False)
        self.bn_c = nn.BatchNorm1d(self.hidden_size)

        self.adj = adj
        self.n_links = adj[0].size(0)
        self.num_weights = adj[0].size(2)
        # self.num_weights = 1

        self.gcn = GCN
        if GCN:
            self.out_features = out_features
            in_features = 1
            self.gconv = TrafficGraphConvolution(self.n_links, self.out_features, out_channels, adj, bias=bias)

        # self.c_weight = Parameter(torch.FloatTensor(self.n_links, self.n_links, self.num_weights))
        # self.cx_reduction = dimReductionOperation(num_weights=self.num_weights, n_neighbor=1, out_channels=1)

        self.bias = bias
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hx, cx):

        if self.gcn:
            # x: [batch size, n_links] --> [batch size, n_links, number of gconv out channel * num_neighbor]
            x = self.gconv(x)
            # x: [...] -->  [batch size, n_links * number of gconv out channel * num_neighbor]
            x = x.reshape(x.size(0), -1)

        gates = self.bn_xh(self.x2h(x)) + self.bn_hh(self.h2h(hx))

        gates = gates.squeeze()

        # gates: [batch size, hidden size] * 4
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = torch.mul(cx, forgetgate) + torch.mul(ingate, cellgate)
        hy = torch.mul(outgate, torch.tanh(self.bn_c(cy)))

        return hy, cy


class Decoder_LSTM(nn.Module):
    """
    An implementation of (MW-TGC) + (Decoder LSTM Cell).
    """

    def __init__(self, output_size, hidden_size, output_dropout=0.5, bias=False):
        super(Decoder_LSTM, self).__init__()

        self.output_size = output_size
        self.hidden_size = hidden_size
        self.x2h = nn.Linear(output_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)

        self.bn_xh = nn.BatchNorm1d(4 * self.hidden_size, affine=False)
        self.bn_hh = nn.BatchNorm1d(4 * self.hidden_size, affine=False)
        self.bn_c = nn.BatchNorm1d(self.hidden_size)

        self.fc = nn.Linear(hidden_size, output_size, bias=bias)
        self.fc_drop = nn.Dropout(output_dropout)

        self.bias = bias
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hx, cx):
        # x: [batch size, hidden size]
        gates = self.bn_xh(self.x2h(x)) + self.bn_hh(self.h2h(hx))

        gates = gates.squeeze()

        # gates: [batch size, hidden size] * 4
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = torch.mul(cx, forgetgate) + torch.mul(ingate, cellgate)
        hy = torch.mul(outgate, torch.tanh(self.bn_c(cy)))

        # generating outputs: next_x = [batch size, N]
        next_x = self.fc(self.fc_drop(hy))

        return next_x, hy, cy


class TrafficGraphConvolution(nn.Module):
    def __init__(self, n_links, out_features, out_channels, adj, dropout=0.3, bias=False, init='uniform'):
        super(TrafficGraphConvolution, self).__init__()
        self.out_features = out_features
        self.num_weights = adj[0].shape[2]
        self.n_neighbor = len(adj)
        self.n_links = n_links

        # weight: N*N weight for each weighted adjacency matrix and for each rank
        self.weight = Parameter(torch.FloatTensor(n_links, n_links, self.num_weights, self.n_neighbor))
        self.adj = torch.stack(adj, 3)
        self.dropout = nn.Dropout(dropout)
        self.dim_reduc = dimReductionOperation(in_channels=self.num_weights * self.n_neighbor, out_channels=out_channels)
        self.bias_bool = bias

        if bias:
            self.bias = Parameter(torch.FloatTensor(self.n_neighbor, self.num_weights))
        else:
            self.register_parameter('bias,', None)

        if init == 'uniform':
            print("| Uniform Initialization")
            self.reset_parameters_uniform()
        elif init == 'xavier':
            print("| Xavier Initialization")
            self.reset_parameters_xavier()
        elif init == 'kaiming':
            print("| Kaiming Initialization")
            self.reset_parameters_kaiming()
        else:
            raise NotImplementedError

    def reset_parameters_uniform(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias_bool:
            self.bias.data.uniform_(-stdv, stdv)

    def reset_parameters_xavier(self):
        nn.init.xavier_normal_(self.weight.data, gain=0.02)  # Implement Xavier Uniform
        if self.bias_bool:
            nn.init.constant_(self.bias.data, 0.0)

    def reset_parameters_kaiming(self):
        nn.init.kaiming_normal_(self.weight.data, a=0, mode='fan_in')
        if self.bias_bool:
            nn.init.constant_(self.bias.data, 0.0)

    def forward(self, x):
        x = self.dropout(x)
        batch_size = x.size(0)

        # element-wise multiplication: weights * adjacency matrices
        weightedADJ = torch.mul(self.adj, self.weight)
        # x: [batch size, N] -> [N, B]
        x = x.permute(1, 0)
        out = []
        for rank in range(self.n_neighbor):
            # reshape weighted matrix
            # gconv: [batch size, n_links, num_weights]
            # [N * number of weights, N] * [N, B] -> [N * # weights, B]
            gconv = weightedADJ[:, :, :, rank].reshape(self.n_links,
                                                       self.n_links * self.num_weights).transpose(0, 1).matmul(x)
            gconv = gconv.transpose(0, 1)
            # gconv: [batch size, n_links, num_weights]
            gconv = gconv.reshape(batch_size, self.n_links, self.num_weights)

            if self.bias is not None:
                gconv = gconv + self.bias[rank, :]

            gconv = F.relu(gconv)

            # out: [batch size, n_links, num_weights * num_neighbor]
            out.append(gconv)

        out = torch.stack(out, len(gconv.size()))
        out = out.view(-1, self.n_links, self.num_weights * self.n_neighbor)
        # final: [batch size, n links, out channels]
        out = self.dim_reduc(out)
        return out

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class dimReductionOperation(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(dimReductionOperation, self).__init__()

        self.conv = nn.Conv2d(in_channels=1, out_channels=out_channels,
                              kernel_size=(1, in_channels), stride=1, padding=0)
        self.bn = nn.BatchNorm1d(out_channels)
        # self.norm = nn.batchNorm2d(out_channels)

    def forward(self, input):
        input = input.unsqueeze(1)
        conv = self.conv(input)
        # norm = self.norm(conv)
        # out = F.relu(norm)

        conv = conv.squeeze(3)

        conv = F.relu(self.bn(conv))
        conv = conv.permute(0, 2, 1)

        return conv
