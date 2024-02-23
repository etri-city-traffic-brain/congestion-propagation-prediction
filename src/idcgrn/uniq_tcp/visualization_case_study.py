import shapefile
import pandas as pd
import os
import folium
import numpy as np
import networkx as nx
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from prettytable import PrettyTable
from traintest_MegaCRN import evaluate, get_model
from pyproj import Proj, transform

# shp_path_node = '../Nodeshape20220614.shp'
shp_path_node = '../Nodeshape20211221.shp'
sf_node = shapefile.Reader(shp_path_node, encoding='cp949')

# shp_path_link = '../Linkshape20220614.shp'
shp_path_link = './Linkshape20211221.shp'
sf_link = shapefile.Reader(shp_path_link, encoding='cp949')

fields_node = [x[0] for x in sf_node.fields[1:]]
records_node = sf_node.records()
shps = [s.points for s in sf_node.shapes()]

fields_link = [x[0] for x in sf_link.fields[1:]]
records_link = sf_link.records()

node_dataframe = pd.DataFrame(columns=fields_node, data=records_node)
node_dataframe = node_dataframe.assign(coords=shps)
link_dataframe = pd.DataFrame(columns=fields_link, data=records_link)

node_dataframe['KEY'] = node_dataframe['IDXNAME'].map(str) + '_' + node_dataframe['NODE_ID'].map(str)

df_link = pd.read_csv('../tmap/DUNSAN/dunsan.csv')
df_link['ST_ND_KEY'] = df_link['IDXNAME'].map(str) + '_' + df_link['ST_ND_ID'].map(str)
df_link['ED_ND_KEY'] = df_link['IDXNAME'].map(str) + '_' + df_link['ED_ND_ID'].map(str)

nodes = node_dataframe[['KEY', 'coords']]
links = df_link[['LINK_ID', 'ST_ND_KEY', 'ED_ND_KEY']]

source_check = links['ST_ND_KEY'].apply(lambda x : x in list(nodes['KEY']))
target_check = links['ED_ND_KEY'].apply(lambda x : x in list(nodes['KEY']))

links = links[source_check & target_check]

G = nx.Graph()
# R is the Earth's radius
R = 6371e3

for idx,row in nodes.iterrows():
    # add node to Graph G
    G.add_node(row['KEY'],Label=row['KEY'],latitude=row['coords'][0][1], longitude=row['coords'][0][0])

for idx,row in links.iterrows():
    ## Calculate the distance between Source and Target Nodes
    lon1 = float(nodes[nodes['KEY'] == row['ST_ND_KEY']]['coords'].values[0][0][0] * np.pi/180)
    lat1 = float(nodes[nodes['KEY'] == row['ST_ND_KEY']]['coords'].values[0][0][1] * np.pi/180)
    lon2 = float(nodes[nodes['KEY'] == row['ED_ND_KEY']]['coords'].values[0][0][0] * np.pi/180)
    lat2 = float(nodes[nodes['KEY'] == row['ED_ND_KEY']]['coords'].values[0][0][1] * np.pi/180)
    d_lat = lat2 - lat1
    d_lon = lon2 - lon1
    a = np.sin(d_lat/2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(d_lon/2) ** 2
    c = 2 * np.arctan2(a**0.5, (1-a) ** 0.5)
    d = R * c

    # Link attribute : 'Source', 'Target' and weight = 'Length between them'
    G.add_edge(row['ST_ND_KEY'], row['ED_ND_KEY'], weight=d)

std_point = tuple(nodes[nodes['KEY'] == '56410000_6016']['coords'].iloc[0][0])
std_point_tuple = (std_point[1], std_point[0])

model = get_model()
modelpt_path = './MegaCRN_20230921140951.pt'
model.load_state_dict(torch.load(modelpt_path))

mean_loss, mae_3, mape_3, rmse_3, mae_6, mape_6, rmse_6, mae_12, mape_12, rmse_12, a_12_60, a_12_75, a_12_90, f_12_60, f_12_75, f_12_90, x_true, y_true, y_pred = evaluate(model, 'case_study')
x_true_process = x_true[0].squeeze()
y_true_process = y_true[:,0,:,:].squeeze()
y_pred_process = y_pred[:,0,:,:].squeeze()

dunsan_speed = pd.read_hdf('../tmap/DUNSAN/dunsan.h5')

dunsan_link_list = dunsan_speed.columns.to_list()
dunsan_link_dict = {i:dunsan_link_list[i] for i in range(len(dunsan_link_list))}

node_list = dunsan_link_list
speed = pd.read_csv('../tmap/DUNSAN/dunsan_link_speedlimit.csv')
avgspeed = dict()
for k in range(len(speed)):
    l, s = speed.iloc[k]
    avgspeed[l] = s
limitSpeed_40 = dict()
limitSpeed_50 = dict()
limitSpeed_60 = dict()
limitSpeed_75 = dict()
limitSpeed_90 = dict()
for id in node_list:
    limitSpeed_40[id] = avgspeed[int(id)] * 0.4
    limitSpeed_50[id] = avgspeed[int(id)] * 0.5
    limitSpeed_60[id] = avgspeed[int(id)] * 0.6
    limitSpeed_75[id] = avgspeed[int(id)] * 0.75
    limitSpeed_90[id] = avgspeed[int(id)] * 0.9

speed_list_40 = list()
speed_list_50 = list()
speed_list_60 = list()
speed_list_75 = list()
speed_list_90 = list()
for key in limitSpeed_60.keys():
    speed_list_40.append(limitSpeed_40[key])
    speed_list_50.append(limitSpeed_50[key])
    speed_list_60.append(limitSpeed_60[key])
    speed_list_75.append(limitSpeed_75[key])
    speed_list_90.append(limitSpeed_90[key])

speed_limit_40 = np.array(speed_list_40).reshape(1, len(node_list))
speed_limit_50 = np.array(speed_list_50).reshape(1, len(node_list))
speed_limit_60 = np.array(speed_list_60).reshape(1, len(node_list))
speed_limit_75 = np.array(speed_list_75).reshape(1, len(node_list))
speed_limit_90 = np.array(speed_list_90).reshape(1, len(node_list))

y_pred_15_congestion = y_pred_process[2,:].cpu().detach().numpy() - speed_limit_40
y_pred_30_congestion = y_pred_process[5,:].cpu().detach().numpy() - speed_limit_40
y_pred_45_congestion = y_pred_process[8,:].cpu().detach().numpy() - speed_limit_40
y_pred_60_congestion = y_pred_process[11,:].cpu().detach().numpy() - speed_limit_40
x_true_60_congestion = x_true_process[11,:].cpu().detach().numpy() - speed_limit_40

# y_pred_congestion <= 0 : 혼잡
# y_pred_congestion > 0 : 원활
y_pred_15_congestion[y_pred_15_congestion <= 0] = 0
y_pred_15_congestion[y_pred_15_congestion > 0] = 1

y_pred_30_congestion[y_pred_30_congestion <= 0] = 0
y_pred_30_congestion[y_pred_30_congestion > 0] = 1

y_pred_45_congestion[y_pred_45_congestion <= 0] = 0
y_pred_45_congestion[y_pred_45_congestion > 0] = 1

y_pred_60_congestion[y_pred_60_congestion <= 0] = 0
y_pred_60_congestion[y_pred_60_congestion > 0] = 1

x_true_60_congestion[x_true_60_congestion <= 0] = 0
x_true_60_congestion[x_true_60_congestion > 0] = 1

# dataframe 생성 ('LINKID_KEY', 'ST_ND_KEY', 'ED_ND_KEY', 'CONGESTION')
linkid = node_list

visualization_list = ['x_true_60', 'y_pred_15', 'y_pred_30', 'y_pred_45', 'y_pred_60']

for visualization_item in visualization_list:
    map_osm = folium.Map(location=std_point_tuple, zoom_start=15)

    if visualization_item == 'x_true_60':
        visualization_data = x_true_60_congestion
    elif visualization_item == 'y_pred_15':
        visualization_data = y_pred_15_congestion
    elif visualization_item == 'y_pred_30':
        visualization_data = y_pred_30_congestion
    elif visualization_item == 'y_pred_45':
        visualization_data = y_pred_45_congestion
    elif visualization_item == 'y_pred_60':
        visualization_data = y_pred_60_congestion

    for ix, row in nodes.iterrows():
        location = (row['coords'][0][1] + 0.003, row['coords'][0][0] - 0.002)  # 위도, 경도 튜플
        folium.Circle(
            location=location,
            # radius=G.degree[row['KEY']] * 30, # 지름이 degree에 비례하도록 설정
            radius=1,
            color='white',
            weight=1,
            fill_opacity=0.6,
            opacity=1,
            fill_color='red',
            fill=True,  # gets overridden by fill_color
            # popup=str(row['Id'])
        ).add_to(map_osm)
        # folium.Marker(location, popup=row['NODE_NAME']).add_to(map_osm)

    kw = {'opacity': 0.5, 'weight': 2}
    # for ix, row in links.iterrows():
    #     start_reverse = tuple(nodes[nodes['KEY']==row['ST_ND_KEY']]['coords'].iloc[0][0])
    #     start = (start_reverse[1], start_reverse[0])
    #     end_reverse = tuple(nodes[nodes['KEY']==row['ED_ND_KEY']]['coords'].iloc[0][0])
    #     end = (end_reverse[1], end_reverse[0])
    #     folium.PolyLine(
    #         locations=[start, end],
    #         color='blue',
    #         line_cap='round',
    #         **kw,
    #     ).add_to(map_osm)
    for i in range(len(linkid)):
        st_nd_key = links[links['LINK_ID'] == linkid[i]]['ST_ND_KEY'].values[0]
        ed_nd_key = links[links['LINK_ID'] == linkid[i]]['ED_ND_KEY'].values[0]

        start_reverse = tuple(nodes[nodes['KEY'] == st_nd_key]['coords'].iloc[0][0])
        start = (start_reverse[1] + 0.003, start_reverse[0] - 0.002)
        end_reverse = tuple(nodes[nodes['KEY'] == ed_nd_key]['coords'].iloc[0][0])
        end = (end_reverse[1] + 0.003, end_reverse[0] - 0.002)

        if visualization_data[0][i] > 0.5:
            color = 'blue'
        elif visualization_data[0][i] < 0.5:
            color = 'red'

        folium.PolyLine(
            locations=[start, end],
            color=color,
            line_cap='round',
            **kw,
        ).add_to(map_osm)

    map_osm.save(f'../visualization_result/{visualization_item}.html')
