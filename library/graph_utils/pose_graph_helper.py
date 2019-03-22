"""
File authors

Christopher Lang
----------------
PoseGraphHelper class

Jens Petit
----------
latLonConvert()
windowSmoother()
deltaAngle()
getGraphsDateLine()
createDays()
"""

import numpy as np
import rtree
from graph_utils import merger
import datetime as dt
import math
import pandas as pd
import networkx as nx
import copy
from graph_utils import pose_graph_nx
from scipy.spatial import cKDTree
from tabulate import tabulate
from graph_utils.db_interface import DBInterface as dbi


class PoseGraphHelper(object):
    """
        This class implements helper functions to navigate within a PoseGraph,
        that however are not essential for to the PoseGraph class.
    """

    def __init__(self, pose_graph):
        self.pose_graph = pose_graph
        self.nodes = pose_graph.getNodeAttributeNumpy( ["x", "y", "door_counter"], astype='pandas')
        self.door_position = self.nodes.loc[(self.nodes["door_counter"]>0), ['x','y']]

    def printGraphProperties(self):

        info = [["# Nodes", self.pose_graph.nx_graph.number_of_nodes()],
                ["# Edges", self.pose_graph.nx_graph.size()],
                ["Edges length [m]", self.pose_graph.nx_graph.size(weight='odo_dist')],
        #        ["Graph radius [m]", nx.radius(self.pose_graph.nx_graph)],
                ["# door events", self.door_position.shape[0]],
               ]

        print(tabulate(info, headers=["Property", "Count"]))

    def get_bay_locations_within_range(self, center_position, range=100):
        """
            
        """
        rel_door_position = copy.deepcopy(self.door_position)
        
        rel_door_position['x'] -= center_position[0]
        rel_door_position['y'] -= center_position[1]
                
        close_bays_idx = (rel_door_position['x']**2 + rel_door_position['y']**2)<range**2
        
        return self.door_position.loc[close_bays_idx, ['x','y']].as_matrix(), \
                self.door_position.loc[close_bays_idx].index.values

    def export_edges_to_txt(self, filename="edgeFile.txt", reduced=True):
        """
        Export pose graph to an txt file, where every line represents one node.
        :param self:
        :param reduced: if set to True, the representation necessary for the RNN github code is exported.
        :return:
        """

        if reduced==True:
            edges = self.pose_graph.nx_graph.edges

            file = open(filename, "w")

            for edge in edges:
                from_node = edge[0]
                to_node = edge[1]

                lat1, lon1 = self.latLonConvert(self.pose_graph.nx_graph.nodes(data='x')[from_node],
                                                self.pose_graph.nx_graph.nodes(data='y')[from_node])

                latk, lonk = self.latLonConvert(self.pose_graph.nx_graph.nodes(data='x')[to_node],
                                                self.pose_graph.nx_graph.nodes(data='y')[to_node])

                file.write("{0}\t{1}\t{2}\t2\t{3}\t{4}\t{5}\t{6}\n".\
                           format(self.pose_graph.getEdgeKey(edge),
                                  self.pose_graph.getNodeKey(from_node), 
                                  self.pose_graph.getNodeKey(to_node),
                                  lat1, lon1,
                                  latk, lonk
                                 )
                           )
            file.close()

        else:
            nx.write_edgelist(nx_graph, filename, comments='#', delimiter='\t', data=True, encoding='utf-8')

    def export_nodes_to_txt(self, filename="nodesFile.txt", reduced=True):
        """
        Export pose graph to an txt file, where every line represents one node.
        The file has header 'Edge_id \t StartNodeID \t EndNodeId 
                                     \t StartNodeLat \t StartNodeLon 
                                     \t EndNodeLat \t EndNodeLon'
        :param self:
        :param reduced: if set to True, the representation necessary for the RNN github code is exported.
        :return:
        """
        file = open(filename, "w")
        nodes = self.pose_graph.nx_graph.nodes(data=True)

        for node in nodes:
            
            lat, lon = self.latLonConvert(node[1]['x'],
                                          node[1]['y'])

            if reduced == True:
                file.write("{0}\t{1}\t{2}\n".format(str(self.pose_graph.getNodeKey(node[0])), lat, lon))
            else:
                file.write(node[0] + str(node[1]) + "\n")

        file.close()


def latLonConvert(x, y, offset=(861570.226138, 5352505.33286)):
    """
    Converts x,y coordinates with to Lat Lon
    :param x:
    :param y:
    :param offset:
    :return:
    """
    lat_divide = 111192.981461485
    lon_divide = 74427.2442617538

    return (y + offset[1]) / lat_divide, (x + offset[0]) / lon_divide 


def windowSmoother(pose_graph, radius, angle):
    """Computes for each node the average position from its neighbors.

    This is an inplace operation. It will change the pose_graph objects node
    positions to the average of the nodes in certain radius around.

    This takes place INPLACE!

    Parameters
    ----------
    pose_graph : PoseGraph object

    radius : float
        The cutoff radius around each point to look for neighbors.

    angle : float
        The difference in angle which is allowed to be a considered as a
        neighbor.

    Return
    ------
    pose_graph : PoseGraph object
    """

    key, positions = pose_graph.getNodeAttributeNumpy(['x', 'y'])
    angles = pose_graph.getNodeAttributeNumpy(['heading'])[1]
    doors = pose_graph.getNodeAttributeNumpy(['door_counter'])[1]

    print("""In window smoother for {} nodes. Might take a
          while...""".format(len(key)))

    tree = cKDTree(positions)

    print("""Finished creating tree....""")

    # numpy array of lists
    key_pos = tree.query_ball_point(positions, r=radius)

    print("""Finished finding NN""")

    node_attributes = {}

    for idx, neighbors in enumerate(key_pos):

        if (idx % 10000 == 0):
            print("on node {} of {}".format(idx, len(key)))

        is_door = bool(doors[idx])

        reference_angle = angles[idx]

        nodes = key[neighbors]

        calc_list = [idx]

        for i, node in enumerate(nodes):

            neighbor_is_door = bool(doors[neighbors[i]])

            if (is_door == neighbor_is_door):
                test_angle = angles[neighbors[i]]

                if (deltaAngle(test_angle, reference_angle) < angle):
                    calc_list.append(neighbors[i])

        mean_pos = np.mean(positions[calc_list], axis=0)

        node_dict = {'x': mean_pos[0], 'y': mean_pos[1]}

        node_attributes.update({key[idx]: node_dict})

    nx.set_node_attributes(pose_graph.nx_graph, node_attributes)

    return pose_graph


def deltaAngle(alpha_1, alpha_2):
    """Returns absolute smaller angle between two angles.

    Parameters
    ----------
    alpha_1 : float
        Angle in radians between -pi and pi.

    alpha_2 : float
        Angle in radians between -pi and pi.

    Returns
    -------
    d_angle : float
        The smaller angle between the two
    """
    d_angle = alpha_1 - alpha_2

    if (d_angle > math.pi):
        d_angle -= 2 * math.pi
    elif (d_angle < -math.pi):
        d_angle += 2 * math.pi

    return abs(d_angle)


def getGraphsDateLine(dates, lines_route, limit_line=100, merged=False):
    """Creates graphs from individual trips specified through the parameters.

    Parameters
    ----------
    line_route : list of tuple (line, route)

    limit_line : int
        Maximum number of trips for one line_route

    merged : Boolean
        Flag indicating if the pulled trips should be merged together into one
        PoseGraph object.

    Returns
    -------
    pose_graph_list : list of PoseGraph objects
        Contains the pulled trips.
    """

    print_string = """Starting to get trip Ids for {} days and {} lines"""

    print(print_string.format(len(dates), len(lines_route)))

    pose_graph_list = []

    trips = []

    test_set = set()

    for date in dates:

        print("Now on date {}.".format(date))

        for line, route in lines_route:
            print("Starting to get trip IDs. Takes a while....")
            trip_ids = dbi.getTripsForRouteAndDay(line, date, route)
            print("We have {} graphs".format(len(trip_ids)))
            trip_ids_cut = trip_ids.head(limit_line)

            trips.extend(list(trip_ids_cut.values.flatten()))

            for trip in trips:
                if not(trip in test_set):
                    test_set.add(trip)
                    graph = pose_graph_nx.PoseGraph()

                    try:
                        graph.loadTripFromDb(trip, line)
                        pose_graph_list.append(graph)
                    except (TypeError, AttributeError, rtree.core.RTreeError):
                        print("NaN data, no interpolation possible")
                        pass

    if (merged):
        print("Start merging them")
        base_graph = pose_graph_list.pop(0)

        for i, graph in enumerate(pose_graph_list):
            if (i % 20 == 0):
                print("Merged {} out of {}".format(i, len(pose_graph_list)))
            try:
                merger.globalXYMerge(base_graph, graph)
            except nx.exception.NetworkXError:
                print("There are duplicate ids, lets jump this graph...")
                pass

        pose_graph_list = base_graph

    return pose_graph_list


def createDays(start_date, days):
    """Creates consecutive datetime strings from a start date.

    Parameters
    ----------

    start_date : string
        In the form YYYY-MM-DD.

    days : int
        How many days from the start_date on to return.

    Returns
    -------
    dates : list of strings
        Contains consecutive dates of strings in a list.
    """

    start_date_dt = dt.datetime.strptime(start_date, "%Y-%m-%d")

    dates = []

    dates.append(start_date)

    for i in range(1, days):
        op_date = start_date_dt + dt.timedelta(days=i)
        dates.append(op_date.strftime("%Y-%m-%d"))

    return dates
