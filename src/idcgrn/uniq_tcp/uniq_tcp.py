import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from prettytable import PrettyTable
from traintest_MegaCRN import evaluate, get_model

parser = argparse.ArgumentParser()
parser.add_argument('--loc', type=str, choices=['DUNSAN'], default='DUNSAN', help='which tmap to run')
parser.add_argument('--pre_train', type=bool, default=True, help='pre_train mode')
parser.add_argument('--modelpt_path', type=str, default='./MegaCRN_20230921140951.pt', help='pre_train mode')

args = parser.parse_args()


def main():
    print('#'*70)
    print('Traffic Congestion Predictor V4.0'.center(70, ' '))
    print('Electronics and Telecommunications Research Institutes'.center(70, ' '))
    print('#'*70)

    if args.pre_train:
        print(f'# Pre-trained model for {args.loc} road network.')
        model = get_model()
        modelpt_path = args.modelpt_path
        model.load_state_dict(torch.load(modelpt_path))
        print(f'# Load pre-trained model complete. (pre-trained model path : {modelpt_path}')
        print(f'# Evaluate the prediction performance on {args.loc} road network ...')
        mean_loss, mae_3, mape_3, rmse_3, mae_6, mape_6, rmse_6, mae_12, mape_12, rmse_12, a_12_60, a_12_75, a_12_90, f_12_60, f_12_75, f_12_90, x_true, y_true, y_pred = evaluate(model, 'test')
        a_12_average = (a_12_60 + a_12_75 + a_12_90) / 3.0
        f_12_average = (f_12_60 + f_12_75 + f_12_90) / 3.0

    data = {
        'Metric': ['Accuracy', 'F1 score'],
        '60%': [a_12_60, f_12_60],
        '75%': [a_12_75, f_12_75],
        '90%': [a_12_90, f_12_90],
        'Average': [a_12_average, f_12_average]
    }
    data_df = pd.DataFrame(data)
    table = PrettyTable()
    table.field_names = data_df.columns
    for row in data_df.itertuples(index=False):
        table.add_row(row)

    print('=' * 70)
    print('Summary of experimental results'.center(70, ' '))
    print('=' * 70)
    print(table)


if __name__ == '__main__':
    main()
