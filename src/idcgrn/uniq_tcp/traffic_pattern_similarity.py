import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from prettytable import PrettyTable
from traintest_MegaCRN import evaluate, get_model, pattern_similarity

parser = argparse.ArgumentParser()
parser.add_argument('--loc', type=str, choices=['DUNSAN'], default='DUNSAN', help='which tmap to run')
parser.add_argument('--pre_train', type=bool, default=True, help='pre_train mode')
parser.add_argument('--modelpt_path', type=str, default='./MegaCRN_20230921140951.pt', help='pre_train mode')
parser.add_argument('--node_index', type=int, default=0, help='0~423 node index')

args = parser.parse_args()


def main():
    print('#'*70)
    print('Traffic Pattern Similarity'.center(70, ' '))
    print('Electronics and Telecommunications Research Institutes'.center(70, ' '))
    print('#'*70)

    if args.pre_train:
        print(f'# Pre-trained model for {args.loc} road network.')
        model = get_model()
        node_index = args.node_index
        modelpt_path = args.modelpt_path
        model.load_state_dict(torch.load(modelpt_path))
        print(f'# Load pre-trained model complete. (pre-trained model path : {modelpt_path}')
        print(f'# Evaluate node {node_index} pattern similarity {args.loc} road network ...')
        attention_score = pattern_similarity(model, 'case_study')

        attention_score = attention_score[node_index].cpu().detach().numpy() * 100
        attention_score = attention_score.tolist()
        attention_score = np.round(attention_score, 2)

        data = {
            'Patterns': ['Pattern 1', 'Pattern 2', 'Pattern 3', 'Pattern 4', 'Pattern 5', 'Pattern 6', 'Pattern 7', 'Pattern 8', 'Pattern 9', 'Pattern 10', 'Pattern 11', 'Pattern 12', 'Pattern 13', 'Pattern 14', 'Pattern 15', 'Pattern 16', 'Pattern 17', 'Pattern 18', 'Pattern 19', 'Pattern 20'],
            'Similarity': attention_score,
        }

        data_df = pd.DataFrame(data)
        table = PrettyTable()
        table.field_names = data_df.columns
        for row in data_df.itertuples(index=False):
            table.add_row(row)

        print('=' * 70)
        print(f'Node {node_index} Pattern Similarity'.center(70, ' '))
        print('=' * 70)
        print(table)


if __name__ == '__main__':
    main()
