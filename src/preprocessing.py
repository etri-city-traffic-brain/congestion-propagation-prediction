import pandas as pd
import numpy as np
import networkx as nx
import argparse


def location(loc):
    """
    :param loc: input location
    :return: location small letter, capital letter
    """
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

def get_mwtgc_weight(loc):
    """
    :param loc: location
    :return: output file. Get the weights used by the mw-tgc model
    """
    loc_kr, loc_en, loc_EN = location(loc)
    df = pd.read_csv('./tmap/%s/Adj(%s_un).csv' %(loc_en, loc_EN), header=None)
    speed_limit = pd.read_csv('tsdlink_avgspeed_20211221.csv')
    file = open('./tmap/%s/%s_node_list.txt' % (loc_en, loc_en))
    node_list = []
    for line in file.readlines():
        node_list.append(line.strip('\n'))

    sl_list = []
    for node in node_list:
        sl_list.append(int(speed_limit.loc[np.where(speed_limit['tsdlinkid'] == int(node))]['avgspeed']))

    adj_r, adj_c = df.shape
    df = df.astype('float')

    for r in range(adj_r):
        for c in range(adj_c):
            print(r,c)
            if df[r][c] == 1:
                sl = sl_list[c] / sl_list[r]
                df[r][c] = sl
                print(sl)

    df.to_csv(f'./tmap/{loc_en}/sl_Adj({loc_EN}_un).csv', index=False, header=None)

    df = pd.read_csv('./tmap/%s/Adj(%s_un).csv' % (loc_en, loc_EN), header=None)

    for r in range(adj_r):
        for c in range(adj_c):
            if df[r][c] == 1:
                if sl_list[c] <= sl_list[r]:
                    slc = sl_list[r]
                else:
                    slc = sl_list[c]
                # slc = sl_list[c] / sl_list[r]
                df[r][c] = slc
                # print(slc)

    df.to_csv(f'./tmap/{loc_en}/slc_Adj({loc_EN}_un).csv', index=False, header=None)

    ## speed limit-change

    df = pd.read_csv('./tmap/%s/Adj(%s_un).csv' % (loc_en, loc_EN), header=None)

    for r in range(adj_r):
        for c in range(adj_c):
            if df[r][c] == 1:
                if sl_list[c] != sl_list[r]:
                    sl_change = 1
                else:
                    sl_change = 0
                # slc = sl_list[c] / sl_list[r]
                df[r][c] = sl_change

    df.to_csv(f'./tmap/{loc_en}/slcha_Adj({loc_EN}_un).csv', index=False, header=None)

    ## plain
    df = pd.read_csv('./tmap/%s/Adj(%s_un).csv' %(loc_en, loc_EN), header=None)
    plain_df = df.dot(df)
    plain_df = plain_df.dot(df)
    plain_df.to_csv(f'./tmap/{loc_en}/pl_Adj({loc_EN}_un).csv', index=False, header=None)


def calculate_length(loc):
    """
    :param loc: location
    :return: output file. Get the weight of distance used by the mw-tgc model.
    """
    selected_link_info = pd.read_csv(f'./get_lengthv3.csv', encoding='utf-8')
    selected_link_info = selected_link_info.astype({'TLINKIDP1':'str', 'TLINKIDN1':'str'})
    loc_kr, loc_en, loc_EN = location(loc)
    f = open('./tmap/%s/%s_node_list.txt' % (loc_en, loc_en))
    node_length = dict()
    for l in f.readlines():
        node = l.strip("\n")
        length = np.sum(selected_link_info.loc[np.where(
            (selected_link_info["TLINKIDP1"] == node + ".0") | (selected_link_info["TLINKIDN1"] == node + ".0"))][
                            'LENGTH'])
        node_length[node] = length
    G = nx.read_gml('./tmap/%s/%s.gml' % (loc_en, loc_en))
    if loc == "도안":
        G.remove_nodes_from(['10491541', '10491542', '10491531', '10491532'])
    distance_list = []
    for u, v in G.edges():
        sum_of_distance = node_length[u] + node_length[v]
        G[u][v]['distance'] = sum_of_distance
        distance_list.append(sum_of_distance)
    distanceDataFrame = nx.to_pandas_adjacency(G, dtype=int, weight='distance')
    distanceDataFrame.to_numpy()
    std = distanceDataFrame.std()

    adj_matrix = np.exp(-np.square(distanceDataFrame / std))

    adj_matrix[adj_matrix < 0.5] = 0
    adj_matrix.to_csv('./tmap/%s/Adj(%s_dist).csv' %(loc_en, loc_EN), header=None, index=None)



def make_adjacency_matrix(loc):
    """
    :param loc: location
    :return: output file. Make the necessary input files in dcrnn, stgcn, and mw-tgc
    """
    loc_en, loc_EN = location(loc)
    df = pd.read_csv('./tmap/%s/%s.csv' % (loc_en, loc_en))

    link_list = set()
    dp1 = df['TLINKIDP1'].unique()
    dp2 = df['TLINKIDN1'].unique()
    dp1 = dp1.astype('str')
    dp2 = dp2.astype('str')

    for item in dp1:
        link_list.add(item)
    for item in dp2:
        link_list.add(item)

    f = open(f'./tmap/{loc_en}/{loc_en}_link_list.txt', 'w')
    for link in link_list:
        f.write(link + "\n")
    f.close()

    G = nx.Graph()

    file = open(f'./tmap/{loc_en}/{loc_en}_link_list.txt')
    lines = file.readlines()

    df = pd.read_csv("tsdlinkturnmaster_20211221.csv")

    # add edge
    for link in lines:
        link = float(link.strip())
        from_list = df[df['fromtsdlinkid'] == link].values.tolist()
        to_list = df[df['totsdlinkid'] == link].values.tolist()
        all_list = from_list + to_list
        for (u, v) in all_list:
            G.add_edge(u, v)
            G.add_edge(v, u)

    # save as gml
    nx.write_gml(G, f'./tmap/{loc_en}/{loc_en}.gml')

    # make adjacency matrix as csv
    A = nx.to_numpy_matrix(G)

    pd.DataFrame(A).to_csv(f'./tmap/{loc_en}/Adj({loc_EN}_un).csv', index=False,header=False)


    with open(f'./tmap/{loc_en}/{loc_en}_node_list.txt', 'w') as f:
        for line in list(G.nodes()):
            f.write(str(line) + "\n")

    ### 노드별 속도
    node_list = list(G.nodes())
    node_list = list(map(str,node_list))
    df = pd.read_csv('07_09.csv')
    loc_df = df[['date'] + node_list]

    loc_df.to_csv(f'./tmap/{loc_en}/{loc_en}_07_09.csv', index=False)

    ### 요일, 시간별로 결측치 대체
    df_07_09 = pd.read_csv(f'./tmap/{loc_en}/{loc_en}_07_09.csv')


    df_07_09['yyyymmdd'] = df_07_09['date'].astype(str).str.slice(start=0, stop=8)
    df_07_09['yyyymmdd2'] = pd.to_datetime(df_07_09['yyyymmdd'])
    df_07_09['day_name'] = df_07_09['yyyymmdd2'].dt.day_name()
    df_07_09 = df_07_09.drop(columns='yyyymmdd2')
    df_07_09 = df_07_09.drop(columns='yyyymmdd')
    df_07_09['time'] = df_07_09['date'].astype(str).str.slice(start=8, stop=12)
    idx_weekend = df_07_09[(df_07_09['day_name'] == 'Saturday') | (df_07_09['day_name'] == 'Sunday')].index
    remove_weekend = df_07_09.drop(idx_weekend)
    remove_weekend = remove_weekend.reset_index()
    g_rw = remove_weekend.groupby(['day_name', 'time']).mean().reset_index()
    numRows, numColumns = remove_weekend.shape

    col = list(remove_weekend.columns)

    for rIdx in range(numRows):
        for cIdx in range(numColumns):
            if remove_weekend.iat[rIdx, cIdx] == 0:
                day = remove_weekend.at[rIdx, 'day_name']
                time = remove_weekend.at[rIdx, 'time']
                node = col[cIdx]
                idx_weekend = g_rw[(g_rw['day_name'] == day) & (g_rw['time'] == time)]
                value = idx_weekend[node].values[0]
                remove_weekend.iat[rIdx, cIdx] = value

    remove_weekend = remove_weekend.replace(0, np.NaN)

    remove_weekend_interpolate = remove_weekend.interpolate(method='linear', limit_direction='backward')
    remove_weekend_interpolate = remove_weekend_interpolate.interpolate(method='linear', limit_direction='forward')

    remove_weekend_interpolate.drop(['index', 'day_name', 'time'], inplace=True, axis=1)
    remove_weekend_interpolate.to_csv(f'./tmap/{loc_en}/{loc_en}_avg_by_weekday_07_09.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--loc', type=str, default='front',help='location ')
    args = parser.parse_args()
    loc = args.loc
    make_adjacency_matrix(loc)
    calculate_length(loc)
    get_mwtgc_weight(loc)


