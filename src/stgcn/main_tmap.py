import dgl
import random
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from load_data import *
from utils import *
from model import *
from sensors2graph import *
import torch.nn as nn
import argparse
import scipy.sparse as sp
import h5py

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


parser = argparse.ArgumentParser(description='STGCN_WAVE')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--disablecuda', action='store_true', help='Disable CUDA')
parser.add_argument('--batch_size', type=int, default=50, help='batch size for training and validation (default: 50)')
parser.add_argument('--epochs', type=int, default=1, help='epochs for training  (default: 50)')
parser.add_argument('--num_layers', type=int, default=9, help='number of layers')
parser.add_argument('--window', type=int, default=144, help='window length')
parser.add_argument('--sensorsfilepath', type=str, default='./data/sensor_graph/graph_sensor_ids.txt', help='sensors file path')
parser.add_argument('--disfilepath', type=str, default='./data/sensor_graph/distances_seoul_2018.csv', help='distance file path')
parser.add_argument('--tsfilepath', type=str, default='./data/seoul-core.h5', help='ts file path')
parser.add_argument('--savemodelpath', type=str, default='stgcnwavemodel.pt', help='save model path')
parser.add_argument('--pred_len', type=int, default=5, help='how many steps away we want to predict')
parser.add_argument('--control_str', type=str, default='TNTSTNTST', help='model strcture controller, T: Temporal Layer, S: Spatio Layer, N: Norm Layer')
parser.add_argument('--channels', type=int, nargs='+', default=[1, 16, 32, 64, 32, 128], help='model strcture controller, T: Temporal Layer, S: Spatio Layer, N: Norm Layer')
parser.add_argument('--loc', type=str, default='front', help='location')
args = parser.parse_args()

device = torch.device("cuda") if torch.cuda.is_available() and not args.disablecuda else torch.device("cpu")

# with open(args.sensorsfilepath) as f:
#     sensor_ids = f.read().strip().split(',')

# distance_df = pd.read_csv(args.disfilepath, dtype={'from': 'str', 'to': 'str'})

# adjacency matrix가 이미 있기 때문에 불러오기만 하면 됨
# adj_mx = get_adjacency_matrix(distance_df, sensor_ids)
loc_en, loc_EN = location(args.loc)

adj_mx = pd.read_csv('../tmap/{}/Adj({}_un).csv'.format(loc_en,loc_EN ), header=None).values

sp_mx = sp.coo_matrix(adj_mx)
G = dgl.from_scipy(sp_mx)

vel = pd.read_csv('../tmap/{}/{}_avg_by_weekday_07_09.csv'.format(loc_en,loc_en))
vel = vel.drop('date', axis=1)

df = pd.DataFrame(vel.to_numpy())
num_samples, num_nodes = df.shape



tsdata = df.to_numpy()


n_his = args.window

save_path = args.savemodelpath



n_pred = args.pred_len
n_route = num_nodes
blocks = args.channels
# blocks = [1, 16, 32, 64, 32, 128]
drop_prob = 0
num_layers = args.num_layers

batch_size = args.batch_size
epochs = args.epochs
lr = args.lr


W = adj_mx
len_val = round(num_samples * 0.1)
len_train = round(num_samples * 0.7)
train = df[: len_train]
val = df[len_train: len_train + len_val]
test = df[len_train + len_val:].to_numpy()

scaler = StandardScaler()
train = scaler.fit_transform(train)
val = scaler.transform(val)
test = scaler.transform(test)
# test2 = scaler.inverse_transform(test1)


x_train, y_train = data_transform(train, n_his, n_pred, device)
x_val, y_val = data_transform(val, n_his, n_pred, device)
x_test, y_test = data_transform(test, n_his, n_pred, device)

train_data = torch.utils.data.TensorDataset(x_train, y_train)
train_iter = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)
val_data = torch.utils.data.TensorDataset(x_val, y_val)
val_iter = torch.utils.data.DataLoader(val_data, batch_size)
test_data = torch.utils.data.TensorDataset(x_test, y_test)
test_iter = torch.utils.data.DataLoader(test_data, batch_size)


loss = nn.MSELoss()
G = G.to(device)
model = STGCN_WAVE(blocks, n_his, n_route, G, drop_prob, num_layers, device, args.control_str).to(device)
optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

min_val_loss = np.inf
for epoch in range(1, epochs + 1):
    l_sum, n = 0.0, 0
    model.train()
    for x, y in train_iter:
        y_pred = model(x).view(len(x), -1)
        l = loss(y_pred, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        l_sum += l.item() * y.shape[0]
        n += y.shape[0]
    scheduler.step()
    val_loss = evaluate_model(model, loss, val_iter)
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        torch.save(model.state_dict(), save_path)
    print("epoch", epoch, ", train loss:", l_sum / n, ", validation loss:", val_loss)

    
best_model = STGCN_WAVE(blocks, n_his, n_route, G, drop_prob, num_layers, device, args.control_str).to(device)
best_model.load_state_dict(torch.load(save_path))

'''
speed limit of each node
'''
node_list = list(vel.columns)
speed_limit = get_limit_speed(node_list, 0.75)
# speed_limit = scaler.transform(speed_limit)
ACC, F1_SCORE = model_accuracy(best_model, test_iter, speed_limit, scaler)
print("ACC:", ACC, ", F1_SCORE:", F1_SCORE)

l = evaluate_model(best_model, loss, test_iter)
MAE, MAPE, RMSE = evaluate_metric(best_model, test_iter, scaler)
print("test loss:", l, "\nMAE:", MAE, ", MAPE:", MAPE, ", RMSE:", RMSE)


