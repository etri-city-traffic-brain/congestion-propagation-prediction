import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def evaluate_model(model, loss, data_iter):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            y_pred = model(x).view(len(x), -1)
            l = loss(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        return l_sum / n


def evaluate_metric(model, data_iter, scaler):
    model.eval()
    with torch.no_grad():
        mae, mape, mse = [], [], []
        for x, y in data_iter:
            y = scaler.inverse_transform(y.cpu().numpy()).reshape(-1)
            y_pred = scaler.inverse_transform(model(x).view(len(x), -1).cpu().numpy()).reshape(-1)
            # y = y.cpu().numpy().reshape(-1)+0.000001
            # y_pred = model(x).view(len(x), -1).cpu().numpy().reshape(-1)
            # y = y.cpu().numpy()
            # y_pred = model(x).view(len(x), -1).cpu().numpy()
            d = np.abs(y - y_pred)
            mae += d.tolist()
            mape += (d / y).tolist()
            mse += (d ** 2).tolist()
        MAE = np.array(mae).mean()
        MAPE = np.array(mape).mean()
        RMSE = np.sqrt(np.array(mse).mean())
        return MAE, MAPE, RMSE


def model_accuracy(model, data_iter, speed_limit, scaler):
    model.eval()
    with torch.no_grad():
        # acc, pre, rec = [], [], []
        acc, f1 = [], []

        for x, y in data_iter:
            y = scaler.inverse_transform(y.cpu().numpy())
            y_pred = scaler.inverse_transform(model(x).view(len(x), -1).cpu().numpy())
            # y = y.cpu().numpy().reshape(-1)+0.000001
            # y_pred = model(x).view(len(x), -1).cpu().numpy().reshape(-1)
            # y = y.cpu().numpy()
            # y_pred = model(x).view(len(x), -1).cpu().numpy()
            # print(len(y))
            for i in range(len(y)):
                y_sub = y[i] - speed_limit
                y_pred_sub = y_pred[i] - speed_limit
                result = y_sub * y_pred_sub
                result[result > 0] = 1
                result[result <= 0] = 0
                acc.append(np.sum(result)/len(speed_limit[0]))

                real_congest = y_sub.copy()
                real_congest[real_congest >= 0] = 0
                real_congest[real_congest < 0] = 1
                pred_congest = y_pred_sub.copy()
                pred_congest[pred_congest >= 0] = 0
                pred_congest[pred_congest < 0] = 1

                # if len(np.unique(real_congest))==1 or len(np.unique(pred_congest))==1:
                #     continue

                f1.append(f1_score(real_congest.tolist(), pred_congest.tolist(), average='micro'))

                # free = np.where(real_congest == 0)
                # congest = np.where(real_congest == 1)
                # TP = np.sum(pred_congest[congest])
                # FN = np.sum(real_congest) - TP
                # FP = np.sum(pred_congest[free])
                # pre.append(TP/(TP+FP))
                # rec.append(TP/(TP+FN))

        ACC = np.array(acc).mean()
        # PRE = np.array(pre).mean()
        # REC = np.array(rec).mean()
        # F1_SCORE = 2 * (PRE*REC) / (PRE+REC)
        F1_SCORE = np.array(f1).mean()
        return ACC, F1_SCORE


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