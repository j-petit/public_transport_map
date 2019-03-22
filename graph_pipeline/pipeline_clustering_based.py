from graph_utils import pose_graph_nx as pg
import pdb
import datetime as dt
from graph_utils.db_interface import DBInterface as dbi
from graph_utils import representor

from graph_utils import merger
from graph_utils import pose_graph_helper

import networkx as nx
import pickle
import glob
import os

import copy

# buslines = ['52', '710', '716', '720', '727', '800', '520', '521', '522',
#             '523', '162', '527', '163', '164', '188', '501', '502', '503',
#             '54', '166', '168', '193', '505', '170', '171', '195', '506', '56',
#             '196', '197', '51', '515', '172', '199', '265', '173', '175',
#             '176', '401', '517', '100', '135', '140', '141', '150', '151',
#             '160', '518', '519', '180', '504', '55', '516', '139', '719', '92',
#             '95', '96', '402', '407', '58', '143', '60', '62', '159', '148',
#             '79', '50', '183', '194', '94', '134', '57', '512', '145', '155',
#             '167', '59', '90', '525', '136', '44', '63', '528', '187', '53',
#             '130', '77', '40', '72', '74', '75', '76', '78', '80', '81', '91',
#             '93', '165', '169', '132', '181', '186', '189', '198', '154',
#             '158', '30', '41', '153', '43', '45', '177', '178', '184', '185',
#             '190', '191', '192', '142', '147', '400', '144', '98', '179',
#             '101', '102', '71', '149', '68', '450', '717']

buslines = ['52', '523', '162', '527', '163', '164', '188', '501', '502', '503',
            '54', '166', '168', '193', '505', '170', '171', '195', '506', '56',
            '196', '197', '51', '515', '172', '199', '265', '173', '175',
            '176', '401', '517', '100', '135', '140', '141', '150', '151',
            '160', '518', '519', '180', '504', '55', '516', '139', '719', '92',
            '95', '96', '402', '407', '58', '143', '60', '62', '159', '148',
            '79', '50', '183', '194', '94', '134', '57', '512', '145', '155',
            '167', '59', '90', '525', '136', '44', '63', '528', '187', '53',
            '130', '77', '40', '72', '74', '75', '76', '78', '80', '81', '91',
            '93', '165', '169', '132', '181', '186', '189', '198', '154',
            '158', '30', '41', '153', '43', '45', '177', '178', '184', '185',
            '190', '191', '192', '142', '147', '400', '144', '98', '179',
            '101', '102', '71', '149', '68', '450', '717']

buslines = [54, 154, 187, 44, 188, 189, 43, 72]
lines_route = [(x, '%') for x in buslines]

path_dump = "/home/data/graphs/pipeline/map_like/herkomerplatz_2/"
start_date = '2018-02-15'

dates = pose_graph_helper.createDays(start_date, 1)

print("Lines {} on dates {}".format(lines_route, dates))

graphs = pose_graph_helper.getGraphsDateLine(dates, lines_route, limit_line=200)

print("We have {} graphs in this map.".format(len(graphs)))
print("Start merging them")

# pickle.dump(graphs, open("save.p", "wb"))

base_graph = graphs.pop(0)

for i, graph in enumerate(graphs):
    if (i % 20 == 0):
        print("Merged {} out of {}".format(i, len(graphs)))
    try:
        merger.globalXYMerge(base_graph, graph)
    except nx.exception.NetworkXError:
        print("There are duplicate ids, lets jump this graph...")
        pass

base_graph.dumpMe(path_dump, name="before_window.graph")
pose_graph_helper.windowSmoother(base_graph, radius=15, angle=1)
base_graph.dumpMe(path_dump, name="after_window.graph")

represent = representor.Representor(base_graph)
represent.computeCentroids(radius=15, angle=1, cutoff_percent=5)
final_map = represent.exportCentroidsInGraph()
represent.intermediate_graph.dumpMe(path_dump, name="intermediate.graph")
final_map.dumpMe(path_dump)
