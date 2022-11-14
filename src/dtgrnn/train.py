from functools import partial
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import dgl
from model import GraphRNN
from dcrnn import DiffConv
from gaan import GatedGAT
from dataloading import METR_LAGraphDataset, METR_LATrainDataset,\
    METR_LATestDataset, METR_LAValidDataset,\
    PEMS_BAYGraphDataset, PEMS_BAYTrainDataset,\
    PEMS_BAYValidDataset, PEMS_BAYTestDataset,\
    SEOUL_GraphDataset, SEOUL_TrainDataset, SEOUL_TestDataset, SEOUL_ValidDataset
from utils import *
import pandas as pd
import scipy.sparse as sp
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

from dataloading import *


def location(loc):
    if loc == 'doan':
        loc_en = 'doan'
        loc_EN = 'Doan'
    elif loc == 'dunsan':
        loc_en = 'dunsan'
        loc_EN = 'Dunsan'
    elif loc == 'wolpyeong':
        loc_en = 'wolpyeong'
        loc_EN = 'Wolpyeong'
    elif loc == 'front':
        loc_en = 'front'
        loc_EN = 'Front'
    return loc_en, loc_EN



batch_cnt = [0]


def train(model, graph, dataloader, optimizer, scheduler, normalizer, loss_fn, device, args):
    total_loss = []
    graph = graph.to(device)
    model.train()
    batch_size = args.batch_size
    for i, (x, y) in enumerate(dataloader):
        optimizer.zero_grad()
        # Padding: Since the diffusion graph is precmputed we need to pad the batch so that
        # each batch have same batch size
        if x.shape[0] != batch_size:
            x_buff = torch.zeros(
                batch_size, x.shape[1], x.shape[2], x.shape[3])
            y_buff = torch.zeros(
                batch_size, x.shape[1], x.shape[2], x.shape[3])
            x_buff[:x.shape[0], :, :, :] = x
            x_buff[x.shape[0]:, :, :,
                   :] = x[-1].repeat(batch_size-x.shape[0], 1, 1, 1)
            y_buff[:x.shape[0], :, :, :] = y
            y_buff[x.shape[0]:, :, :,
                   :] = y[-1].repeat(batch_size-x.shape[0], 1, 1, 1)
            x = x_buff
            y = y_buff
        # Permute the dimension for shaping
        x = x.permute(1, 0, 2, 3)
        y = y.permute(1, 0, 2, 3)

        temp = normalizer.normalize(x).float().to(device)
        x_norm = normalizer.normalize(x).reshape(
            x.shape[0], -1, x.shape[3]).float().to(device)
        y_norm = normalizer.normalize(y).reshape(
            x.shape[0], -1, x.shape[3]).float().to(device)
        y = y.reshape(y.shape[0], -1, y.shape[3]).float().to(device)

        batch_graph = dgl.batch([graph]*batch_size)
        output = model(batch_graph, x_norm, y_norm, batch_cnt[0], device)
        # Denormalization for loss compute
        y_pred = normalizer.denormalize(output)
        loss = loss_fn(y_pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        if get_learning_rate(optimizer) > args.minimum_lr:
            scheduler.step()
        total_loss.append(float(loss))
        batch_cnt[0] += 1
        # print("Batch: ", i)
    return np.mean(total_loss)


def eval(model, graph, dataloader, normalizer, loss_fn, device, args):
    total_loss = []
    graph = graph.to(device)
    model.eval()
    batch_size = args.batch_size
    for i, (x, y) in enumerate(dataloader):
        # Padding: Since the diffusion graph is precmputed we need to pad the batch so that
        # each batch have same batch size
        if x.shape[0] != batch_size:
            x_buff = torch.zeros(
                batch_size, x.shape[1], x.shape[2], x.shape[3])
            y_buff = torch.zeros(
                batch_size, x.shape[1], x.shape[2], x.shape[3])
            x_buff[:x.shape[0], :, :, :] = x
            x_buff[x.shape[0]:, :, :,
                   :] = x[-1].repeat(batch_size-x.shape[0], 1, 1, 1)
            y_buff[:x.shape[0], :, :, :] = y
            y_buff[x.shape[0]:, :, :,
                   :] = y[-1].repeat(batch_size-x.shape[0], 1, 1, 1)
            x = x_buff
            y = y_buff
        # Permute the order of dimension
        x = x.permute(1, 0, 2, 3)
        y = y.permute(1, 0, 2, 3)

        x_norm = normalizer.normalize(x).reshape(
            x.shape[0], -1, x.shape[3]).float().to(device)
        y_norm = normalizer.normalize(y).reshape(
            x.shape[0], -1, x.shape[3]).float().to(device)
        y = y.reshape(x.shape[0], -1, x.shape[3]).to(device)

        batch_graph = dgl.batch([graph]*batch_size)
        output = model(batch_graph, x_norm, y_norm, i, device)
        y_pred = normalizer.denormalize(output)
        loss = loss_fn(y_pred, y)
        total_loss.append(float(loss))
    return np.mean(total_loss)

# def evaluate_metric(model, graph, dataloader, normalizer, loss_fn, device, args):
#     model.eval()
#     with torch.no_grad():
#         mae, mape, mse = [], [], []
#         batch_size = args.batch_size
#         for i, (x, y) in enumerate(dataloader):
#             # Padding: Since the diffusion graph is precmputed we need to pad the batch so that
#             # each batch have same batch size
#             if x.shape[0] != batch_size:
#                 x_buff = torch.zeros(
#                     batch_size, x.shape[1], x.shape[2], x.shape[3])
#                 y_buff = torch.zeros(
#                     batch_size, x.shape[1], x.shape[2], x.shape[3])
#                 x_buff[:x.shape[0], :, :, :] = x
#                 x_buff[x.shape[0]:, :, :,
#                 :] = x[-1].repeat(batch_size - x.shape[0], 1, 1, 1)
#                 y_buff[:x.shape[0], :, :, :] = y
#                 y_buff[x.shape[0]:, :, :,
#                 :] = y[-1].repeat(batch_size - x.shape[0], 1, 1, 1)
#                 x = x_buff
#                 y = y_buff
#             # Permute the order of dimension
#             x = x.permute(1, 0, 2, 3)
#             y = y.permute(1, 0, 2, 3)
#
#             x_norm = normalizer.normalize(x).reshape(
#                 x.shape[0], -1, x.shape[3]).float().to(device)
#             y_norm = normalizer.normalize(y).reshape(
#                 x.shape[0], -1, x.shape[3]).float().to(device)
#             # x_norm = x.reshape(x.shape[0], -1, x.shape[3]).float().to(device)
#             # y_norm = y.reshape(x.shape[0], -1, x.shape[3]).float().to(device)
#             y = y.reshape(x.shape[0], -1, x.shape[3]).to(device)
#             # y = y.cpu().numpy().reshape(-1)
#
#             batch_graph = dgl.batch([graph] * batch_size)
#             output = model(batch_graph, x_norm, y_norm, i, device)
#             # y_pred = normalizer.denormalize(output).cpu().numpy().reshape(-1)
#             # y_pred = model(x).view(len(x), -1).cpu().numpy().reshape(-1)
#             y_pred = normalizer.denormalize(output)
#             # y_pred = output
#             # d = loss_fn(y_pred, y).cpu().numpy()
#             # y = y.cpu().numpy()+0.000001
#
#             mask = (y != 0).float()
#             mask /= mask.mean()
#             d = loss_fn(y_pred, y)
#             # d = torch.abs(y_pred - y)
#             # d = d * mask
#             # d[d != d] = 0
#
#             y = y + 0.000001
#
#             # mae.append(d)
#             # mape.append(d / y)
#             # mse.append(d ** 2)
#             mae += d.tolist()
#             mape += (d / y).tolist()
#             mse += (d ** 2).tolist()
#
#         MAE = np.array(mae).mean()
#         MAPE = np.array(mape).mean()
#         RMSE = np.sqrt(np.array(mse).mean())
#
#         return MAE, MAPE, RMSE
def evaluate_metric(model, graph, dataloader, normalizer, loss_fn, device, args):
    model.eval()
    with torch.no_grad():
        mae, mape, mse = [], [], []
        batch_size = args.batch_size
        for i, (x, y) in enumerate(dataloader):
            # Padding: Since the diffusion graph is precmputed we need to pad the batch so that
            # each batch have same batch size
            if x.shape[0] != batch_size:
                x_buff = torch.zeros(
                    batch_size, x.shape[1], x.shape[2], x.shape[3])
                y_buff = torch.zeros(
                    batch_size, x.shape[1], x.shape[2], x.shape[3])
                x_buff[:x.shape[0], :, :, :] = x
                x_buff[x.shape[0]:, :, :,
                :] = x[-1].repeat(batch_size - x.shape[0], 1, 1, 1)
                y_buff[:x.shape[0], :, :, :] = y
                y_buff[x.shape[0]:, :, :,
                :] = y[-1].repeat(batch_size - x.shape[0], 1, 1, 1)
                x = x_buff
                y = y_buff
            # Permute the order of dimension
            x = x.permute(1, 0, 2, 3)
            y = y.permute(1, 0, 2, 3)

            x_norm = normalizer.normalize(x).reshape(
                x.shape[0], -1, x.shape[3]).float().to(device)
            y_norm = normalizer.normalize(y).reshape(
                x.shape[0], -1, x.shape[3]).float().to(device)
            # x_norm = x.reshape(x.shape[0], -1, x.shape[3]).float().to(device)
            # y_norm = y.reshape(x.shape[0], -1, x.shape[3]).float().to(device)
            y = y.reshape(x.shape[0], -1, x.shape[3]).to(device)
            # y = y.cpu().numpy().reshape(-1)

            batch_graph = dgl.batch([graph] * batch_size)
            output = model(batch_graph, x_norm, y_norm, i, device)
            # y_pred = normalizer.denormalize(output).cpu().numpy().reshape(-1)
            # y_pred = model(x).view(len(x), -1).cpu().numpy().reshape(-1)
            y_pred = normalizer.denormalize(output)
            # y_pred = output
            # d = loss_fn(y_pred, y).cpu().numpy()
            # y = y.cpu().numpy()+0.000001

            mask = (y != 0).float()
            mask /= mask.mean()
            d = loss_fn(y_pred, y)
            # d = torch.abs(y_pred - y)
            # d = d * mask
            # d[d != d] = 0

            y = y + 0.000001
            mae.append(d.cpu())
            mape.append((d.cpu() / y.cpu())[:,:,0].tolist())
            mse.append(d.cpu() ** 2)
            # mae += d.tolist()
            # mape += (d / y).tolist()
            # mse += (d ** 2).tolist()

        MAE = np.array(mae).mean()
        # print(type(mape), mape)
        MAPE = np.array(mape).mean()
        # MAE = np.array(np.array(mae).mean())
        # MAPE = np.array(np.array(mape).mean())
        RMSE = np.sqrt(np.array(mse).mean())

        return MAE, MAPE, RMSE

def model_accuracy(model, graph, dataloader, normalizer, loss_fn, device, args, speed_limit):
    model.eval()
    with torch.no_grad():
        acc, f1 = [], []
        batch_size = args.batch_size
        for i, (x, y) in enumerate(dataloader):
            # Padding: Since the diffusion graph is precmputed we need to pad the batch so that
            # each batch have same batch size
            if x.shape[0] != batch_size:
                x_buff = torch.zeros(
                    batch_size, x.shape[1], x.shape[2], x.shape[3])
                y_buff = torch.zeros(
                    batch_size, x.shape[1], x.shape[2], x.shape[3])
                x_buff[:x.shape[0], :, :, :] = x
                x_buff[x.shape[0]:, :, :,
                :] = x[-1].repeat(batch_size - x.shape[0], 1, 1, 1)
                y_buff[:x.shape[0], :, :, :] = y
                y_buff[x.shape[0]:, :, :,
                :] = y[-1].repeat(batch_size - x.shape[0], 1, 1, 1)
                x = x_buff
                y = y_buff
            # Permute the order of dimension
            x = x.permute(1, 0, 2, 3)
            y = y.permute(1, 0, 2, 3)

            # x_norm = normalizer.normalize(x).float().to(device)
            # y_norm = normalizer.normalize(y).float().to(device)
            x_norm = normalizer.normalize(x).reshape(
                x.shape[0], -1, x.shape[3]).float().to(device)
            y_norm = normalizer.normalize(y).reshape(
                x.shape[0], -1, x.shape[3]).float().to(device)
            # x_norm = x.reshape(x.shape[0], -1, x.shape[3]).float().to(device)
            # y_norm = y.reshape(x.shape[0], -1, x.shape[3]).float().to(device)
            # y = y.reshape(x.shape[0], -1, x.shape[3]).to(device)
            # y = y.cpu().numpy().reshape(-1)

            batch_graph = dgl.batch([graph] * batch_size)
            output = model(batch_graph, x_norm, y_norm, i, device)
            # y_pred = normalizer.denormalize(output).cpu().numpy().reshape(-1)
            # y_pred = model(x).view(len(x), -1).cpu().numpy().reshape(-1)
            y_pred = normalizer.denormalize(output).reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
            limit1 = np.empty((64,len(speed_limit[0])))
            for j in range(64):
                limit1[j] = speed_limit
            limit2 = np.empty((12,64,len(speed_limit[0])))
            for k in range(12):
                limit2[k] = limit1
            limit2 = limit2.reshape((12,64,len(speed_limit[0]),1))
            y = y.cpu().numpy()
            y_pred = y_pred.cpu().numpy()
            new_y = np.delete(y, -1, axis=3)
            new_y_pred = np.delete(y_pred, -1, axis=3)

            y_sub = new_y - limit2
            y_pred_sub = new_y_pred - limit2
            result = y_sub * y_pred_sub
            result[result > 0] = 1
            result[result <= 0] = 0
            acc.append(np.sum(result) / result.size)

            real_congest = y_sub.copy()
            real_congest[real_congest >= 0] = 0
            real_congest[real_congest < 0] = 1
            pred_congest = y_pred_sub.copy()
            pred_congest[pred_congest >= 0] = 0
            pred_congest[pred_congest < 0] = 1

            f1.append(f1_score(real_congest.reshape((-1)), pred_congest.reshape((-1)), average='micro'))

        ACC = np.array(acc).mean()
        F1_SCORE = np.array(f1).mean()
        return ACC, F1_SCORE

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Define the arguments
    parser.add_argument('--batch_size', type=int, default=64,
                        help="Size of batch for minibatch Training")
    parser.add_argument('--num_workers', type=int, default=0,
                        help="Number of workers for parallel dataloading")
    parser.add_argument('--model', type=str, default='dcrnn',
                        help="WHich model to use DCRNN vs GaAN")
    parser.add_argument('--gpu', type=int, default=0,
                        help="GPU indexm -1 for CPU training")
    parser.add_argument('--diffsteps', type=int, default=2,
                        help="Step of constructing the diffusiob matrix")
    parser.add_argument('--num_heads', type=int, default=2,
                        help="Number of multiattention head")
    parser.add_argument('--decay_steps', type=int, default=2000,
                        help="Teacher forcing probability decay ratio")
    parser.add_argument('--lr', type=float, default=0.01,
                        help="Initial learning rate")
    parser.add_argument('--minimum_lr', type=float, default=2e-6,
                        help="Lower bound of learning rate")
    parser.add_argument('--dataset', type=str, default='tmap',
                        help="dataset LA for METR_LA; BAY for PEMS_BAY")
    parser.add_argument('--epochs', type=int, default=10,
                        help="Number of epoches for training")
    parser.add_argument('--max_grad_norm', type=float, default=5.0,
                        help="Maximum gradient norm for update parameters")

    parser.add_argument('--savemodelpath', type=str, default='dcrnnmodel.pt',
                        help='save model path')
    parser.add_argument('--loc', type=str, default='front',
                        help='location')
    args = parser.parse_args()
    # Load the datasets


    loc_en, loc_EN = location(args.loc)

    if args.dataset == 'LA':
        g = METR_LAGraphDataset()
        train_data = METR_LATrainDataset()
        test_data = METR_LATestDataset()
        valid_data = METR_LAValidDataset()
    elif args.dataset == 'BAY':
        g = PEMS_BAYGraphDataset()
        train_data = PEMS_BAYTrainDataset()
        test_data = PEMS_BAYTestDataset()
        valid_data = PEMS_BAYValidDataset()
    elif args.dataset == 'seoul':
        g = SEOUL_GraphDataset()
        train_data = SEOUL_TrainDataset()
        test_data = SEOUL_TestDataset()
        valid_data = SEOUL_ValidDataset()
    elif args.dataset == 'tmap':
        g = TMAP_GraphDataset(args.loc,loc_EN)
        train_data = TMAP_TrainDataset(args.loc)
        test_data = TMAP_TestDataset(args.loc)
        valid_data = TMAP_ValidDataset(args.loc)


    if args.gpu == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(args.gpu))

    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    valid_loader = DataLoader(
        valid_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    test_loader = DataLoader(
        test_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    # print("traindate ===========================================",train_data)
    normalizer = NormalizationLayer(train_data.mean, train_data.std)

    if args.model == 'dcrnn':
        batch_g = dgl.batch([g]*args.batch_size).to(device)
        out_gs, in_gs = DiffConv.attach_graph(batch_g, args.diffsteps)
        net = partial(DiffConv, k=args.diffsteps,
                      in_graph_list=in_gs, out_graph_list=out_gs)
    elif args.model == 'gaan':
        net = partial(GatedGAT, map_feats=64, num_heads=args.num_heads)

    dcrnn = GraphRNN(in_feats=2,
                     out_feats=64,
                     seq_len=12,
                     num_layers=2,
                     net=net,
                     decay_steps=args.decay_steps).to(device)

    optimizer = torch.optim.Adam(dcrnn.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    loss_fn = masked_mae_loss
    min_val_loss = np.inf
    save_path = args.savemodelpath

    for e in range(args.epochs):
        # print(type(dcrnn), type(g), type(train_loader), type(optimizer), type(scheduler))
        # print(type(normalizer), type(loss_fn), type(device), type(args))

        train_loss = train(dcrnn, g, train_loader, optimizer, scheduler,
                           normalizer, loss_fn, device, args)
        valid_loss = eval(dcrnn, g, valid_loader,
                          normalizer, loss_fn, device, args)
        if valid_loss < min_val_loss:
            min_val_loss = valid_loss
            torch.save(dcrnn.state_dict(), save_path)
        test_loss = eval(dcrnn, g, test_loader,
                         normalizer, loss_fn, device, args)
        print("Epoch: {} Train Loss: {} Valid Loss: {} Test Loss: {}".format(e,
                                                                             train_loss,
                                                                             valid_loss,
                                                                             test_loss))

    best_model = GraphRNN(in_feats=2,
                        out_feats=64,
                        seq_len=12,
                        num_layers=2,
                        net=net,
                        decay_steps=args.decay_steps).to(device)

    best_model.load_state_dict(torch.load(save_path))
    node_list = []
    # f = open('../tmap/{}/{}_node_list.txt'.format(args.loc, args.loc))
    # for line in f.readlines():
    #     node_list.append(line.strip('\n'))
    # speed_limit = get_limit_speed(node_list, 0.6)
    # ACC1, f1_1 = model_accuracy(best_model, g, test_loader, normalizer, loss_fn, device, args, speed_limit)
    # print("--------------------0.6-----------------------")
    # print("accuracy",ACC1,"f1 score", f1_1)
    # print()
    # speed_limit = get_limit_speed(node_list, 0.75)
    # acc2, f1_2 = model_accuracy(best_model, g, test_loader, normalizer, loss_fn, device, args, speed_limit)
    # print("===============0.75====================")
    # print("accuracy", acc2, "f1 score", f1_2)
    # print()
    # print("===============0.9================")
    # speed_limit = get_limit_speed(node_list, 0.9)
    # ACC, F1_SCORE = model_accuracy(best_model, g, test_loader, normalizer, loss_fn, device, args, speed_limit)
    # print("ACC:", ACC, ", F1_SCORE:", F1_SCORE)
    MAE, MAPE, RMSE = evaluate_metric(best_model, g, test_loader, normalizer, loss_fn, device, args)
    print("MAE:", MAE, ", MAPE:", MAPE, ", RMSE:", RMSE)
