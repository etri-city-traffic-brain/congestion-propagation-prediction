import shapefile
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import os
import csv


##############################################
# Step 0. perform spatial filtering
sf = shapefile.Reader('./Linkshape20211221.sh')
f_info = sf.fields

###############################################
# MBR: minimum bounding box for spatial fitering
minX = 127.321000
minY = 36.327008
maxX = 127.359796 
maxY = 36.368395

bbox = [minX, minY, maxX, maxY]
fields = ['TLINKIDP1', 'TLINKIDN1', 'LENGTH']

tlink_set = set()   

for shapeRec in sf.iterShapeRecords(bbox=bbox, fields=fields):
    # print(shapeRec.record)
    if shapeRec.record['TLINKIDP1']:
        tlink_set.add(shapeRec.record['TLINKIDP1'])
    if shapeRec.record['TLINKIDN1']:
        tlink_set.add(shapeRec.record['TLINKIDN1'])

# print(tlink_set)
# print('Link set size: ', len(tlink_set))
print('# Completed tsdlink set construction.')


tlink_list = list(tlink_set)
dict = {}
for i in range (0, len(tlink_list)):
    tsdlink_length = 0 
    for shapeRec in sf.iterShapeRecords(bbox=bbox, fields=fields):
        key = tlink_list[i]
        if shapeRec.record['TLINKIDP1'] == key or shapeRec.record['TLINKIDN1'] == key:
            # print(shapeRec.record)
            tsdlink_length += int(shapeRec.record['LENGTH'])
    dict[key] = tsdlink_length            
    # print("Total length: ", tsdlink_length)
    # print(dict)

print("# Completed tsdlink length compuation.")

print(dict)
