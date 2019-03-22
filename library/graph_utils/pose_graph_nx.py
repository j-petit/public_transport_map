"""
File authors:

Jens Petit
----------
all functions except ones stated below

Christopher Lang
----------------
projectOntoEdge()
getEdgeKey()
getNodeKey()
getNodeHeading()
setAllHeading()
calculateEdgeDistances()
updateKDTree()

Georg Aures
-----------
serial_node_drop()
"""

import networkx as nx
import numpy as np
import copy
import pdb
import pandas as pd
from graph_utils import db_interface
from graph_utils import rtree
from slam_utils import veh_localizer
import math
import os
from scipy.spatial import cKDTree, distance
import pickle
import pprint
from collections import Counter


class PoseGraph(object):

    """Implements general class for the pose graph.

    The main datastructure is a DiGraph from the networkx package. All the
    graph data is stored in it.

    Attributes
    ----------
    graph : networkx DiGraph
        See additional description below.

    offset : tuple(x,y) of floats
        Marienplatz coordinates as x,y to substract and shift all values.

    lat_lon_divide : tuple(lat_divide, lon_divide) of floats

    edge_keys : dict

    node_keys : dict

    rtree : RTreeIndexer

    kdtree : cKDTree
        For NN search.


    Networkx DiGraph
    ----------------

    Graph attributes
    ----------------
    trips : set() of ints
        Contains all trip ids which are incorporated into the graph.

    it_counter : int
        Counter how many optimization iterations have run on this graph.

    op_dates : set() of strings

    lines : set() of ints
        Which line numbers are already used in this graph.


    Node attributes
    ---------------
    x : float
        x coordinate corresponding to longitude.

    y : float
        y coordinate corresponding to latitude.

    'gps_x' : float
        x coordinate of the gps measurement.

    'gps_y' : float
        y coordinate of the gps measurement.

    'gps_var_x' : float
        Variance of the gps coordinate in x direction.

    'gps_var_y' : float
        Variance of the gps coordinate in y direction.

    'gps_consistency' : float (0,1)
        Number indicates the initial consistency between the gps and odometry.
        We assume for good measurements that delta GPS should be nearly equal
        to an odometry measurement.

    'door_counter' : Counter()
        Number of door events for a specific line.

    'gps_counter' : float
        Exponential moving count for then unifying two node distributions.

    'gps_quality' : float
        Indicates the quality of GPS in range 0 for no signal, over 1 for bad
        signal, to 2 for good signal.

    Edge attributes
    ---------------
    'odo_dist' : float
        Spatial 1D constraint which gives a euclidean distance for connecting
        two pose nodes. An exponential moving average is used for adding new
        samples to the value.

    'odo_counter' : float
        Exponential moving count to regard the number of samples that are
        already incorporated into the 'odo_dist' attribute for later unifying
        two edge distributions.

    'odo_var' : float
        Variance of the 'odo_dist' value calculated through exponential moving
        variance filter.

    'travel_time' : float
        Delta time in [s] to travel on the edge.
    """

    def __init__(self):
        self.nx_graph = nx.DiGraph()
        self.nx_graph.graph.update({'trips': [],
                                    'it_counter': 0,
                                    'op_dates': set(),
                                    'lines': set()})

        self.offset = (861570.226138, 5352505.33286)  # (x,y)
        self.lat_lon_divide = (111192.981461485, 74427.2442617538)  # (y,x)

        self.edge_keys = {}
        self.node_keys = {}

        self.rtree = rtree.RTreeIndexer(self)
        self.kdtree = None

    def init_with_nx_graph(self, set_to_nx_graph, format="tangential_plane"):
        set_graph = nx.convert_node_labels_to_integers(set_to_nx_graph)
        if (format == "WSG84"):
            for node in set_graph.nodes:
                rescale_x = (float(set_graph.nodes[node]['lon']) *
                             self.lat_lon_divide[1])

                set_graph.nodes[node]['x'] = rescale_x - self.offset[0]

                rescale_y = (float(set_graph.nodes[node]['lat']) *
                             self.lat_lon_divide[0])

                set_graph.nodes[node]['y'] = rescale_y - self.offset[1]

                set_graph.nodes[node]['gps_consistency'] = 1.0

        self.nx_graph = set_graph

    def loadTripFromDb(self, VEH_TRIP_ID, line=None, raw=False, min_sparse=5):
        """Loads the data from a specific trip into the graph.

        The data is written into the nx graph datastructure.

        Parameters
        ----------
        VEH_TRIP_ID : int
            Number in SQL database specifying the trip

        raw : Boolean
            Flag if interpolation used when loading

        Returns
        -------
        Boolean : true if succes, false if fail

        """

        self.nx_graph.graph['lines'].add(line)

        interface = db_interface.DBInterface()

        try:
            data = interface.loadTrip(VEH_TRIP_ID, raw=raw,
                                      min_sparse=min_sparse)

            self.nx_graph.graph['trips'].append(VEH_TRIP_ID)

        except (TypeError):
            print("NaN data, no interpolation possible")
            data = None

        if (data.empty):
            print("No data for this trip {}!".format(VEH_TRIP_ID))
            return False
        else:
            self._loadTrip(data)
            return True

    def _loadTrip(self, data):
        """From a pandas df loads the data.

        The df has to hold all information as provided by the interface to
        the SQL database. The information is then written into the networkx
        graph (directed) datastructer which contains almost all of the data.

        Parameters
        ----------
        data : pandas df
            Has to have columns 'VEH_X', 'VEH_Y',
                                'DOOR',
                                'VEH_GPS_ID',
                                'D_VEH_ODOMETRY',
                                'GPS_CONSISTENCY',
                                'GPS_QUALITY'.
        """

        edges = pd.DataFrame(columns=['from_id', 'to_id',
                                      'odo_dist', 'odo_var',
                                      'num_vehicles',
                                      'odo_counter',
                                      'travel_time',
                                      'euclidean_dist',
                                      'Counter'])

        edges['odo_dist'] = data['D_VEH_ODOMETRY']

        edges['from_id'] = data['VEH_GPS_ID']
        edges['to_id'] = np.roll(data['VEH_GPS_ID'].values, -1)

        edges['travel_time'] = data['D_TIME']

        edges['num_vehicles'] = 1

        edges['odo_counter'] = 1
        edges['odo_var'] = 1
        edges['Counter'] = 1

        edges['euclidean_dist'] = 0

        # Drop NaN exists
        edges = edges[pd.notnull(edges['odo_dist'])]

        G = nx.convert_matrix.from_pandas_edgelist(edges,
                                                   source='from_id',
                                                   target='to_id',
                                                   edge_attr=True,
                                                   create_using=nx.DiGraph())

        line = next(iter(self.nx_graph.graph['lines']))

        door_counter_list = [Counter({line: x}) if x > 0 else Counter() for x
                             in data['DOOR'].values]

        door_counter = pd.Series(door_counter_list,
                                 index=data['VEH_GPS_ID']).to_dict()

        data['VEH_X'] = data['VEH_X'] - self.offset[0]
        data['VEH_Y'] = data['VEH_Y'] - self.offset[1]

        x = pd.Series(data['VEH_X'].values,
                      index=data['VEH_GPS_ID']).to_dict()
        y = pd.Series(data['VEH_Y'].values,
                      index=data['VEH_GPS_ID']).to_dict()

        gps_x = pd.Series(data['VEH_X'].values,
                          index=data['VEH_GPS_ID']).to_dict()
        gps_y = pd.Series(data['VEH_Y'].values,
                          index=data['VEH_GPS_ID']).to_dict()
        gps_consistency = pd.Series(data['GPS_CONSISTENCY'].values,
                                    index=data['VEH_GPS_ID']).to_dict()
        gps_quality = pd.Series(data['GPS_QUALITY'].values,
                                index=data['VEH_GPS_ID']).to_dict()
        gps_var_x = 0.963172827
        gps_var_y = 0.963172827
        gps_counter = 1

        nx.set_node_attributes(G, values=x, name='x')
        nx.set_node_attributes(G, values=y, name='y')
        nx.set_node_attributes(G, values=gps_x, name='gps_x')
        nx.set_node_attributes(G, values=gps_y, name='gps_y')
        nx.set_node_attributes(G, values=1, name='heading')
        nx.set_node_attributes(G, values=gps_var_x, name='gps_var_x')
        nx.set_node_attributes(G, values=gps_var_y, name='gps_var_y')
        nx.set_node_attributes(G, values=gps_consistency,
                               name='gps_consistency')
        nx.set_node_attributes(G, values=door_counter, name='door_counter')
        nx.set_node_attributes(G, values=gps_counter, name='gps_counter')
        nx.set_node_attributes(G, values=gps_quality, name='gps_quality')

        self.nx_graph = nx.algorithms.operators.binary.union(self.nx_graph,
                                                             G)
        self.setAllHeading()
        self.updateEdgeKeys()
        self.rtree.build_r_tree(self)

    def loadNxGraph(self, filename):
        self.nx_graph = nx.read_gpickle(filename)
        self.updateEdgeKeys()
        self.rtree.build_r_tree(self)

    def serial_node_drop(self, node_drop):
        """Drops specified node under assumption that is only has one
        predecessor and successor.

        Parameters
        ----------
        node_drop : int
            ID of the node to drop.
        """

        if (len(self.nx_graph.in_edges(node_drop)) == 1 and
                len(self.nx_graph.out_edges(node_drop)) == 1):

            edge_in, *_ = self.nx_graph.in_edges(node_drop)
            edge_out, *_ = self.nx_graph.out_edges(node_drop)

            if edge_in[0] == edge_out[1]:
                self.nx_graph.remove_node(node_drop)
            else:
                odo_var_sum = (self.nx_graph.edges[edge_in]['odo_var'] +
                               self.nx_graph.edges[edge_out]['odo_var'])
                odo_dist_sum = (self.nx_graph.edges[edge_in]['odo_dist'] +
                                self.nx_graph.edges[edge_out]['odo_dist'])

                travel_time = (self.nx_graph.edges[edge_in]['travel_time'] +
                               self.nx_graph.edges[edge_out]['travel_time'])

                sub_edges = (self.nx_graph.edges[edge_in].get('sub_edges', [edge_in]) +
                             self.nx_graph.edges[edge_out].get('sub_edges', [edge_out]))

                self.nx_graph.add_edge(edge_in[0],
                                       edge_out[1],
                                       odo_var=odo_var_sum,
                                       odo_dist=odo_dist_sum,
                                       sub_edges=sub_edges,
                                       travel_time=travel_time)

                self.nx_graph.remove_node(node_drop)
        else:
            print("Error: number of out and or in edges is not 1")

    def getNodeAttributeNumpy(self, attributes, astype='numpy'):
        """Returns the attribute data of all nodes in standard order.

        All node attributes are written into a single numpy array for further
        processing the data in other applications. It can be rewritten to the
        graph through set_node_attributes of networkx through creating a dict
        from keys and values.

        Parameters
        ----------
        attributes : list of strings
            Attribute names as a list.

        Returns
        -------
        keys: np.array, shape(N)
            The node keys as numpy array where N is the number of nodes.

        np_data : np.array, shape(N,M)
            Where N is the number of nodes and M the number of attributes.
        """
        if astype == 'pandas':

            df = pd.DataFrame()  # create empty data frame

            for attribute in attributes:
                if attribute == 'door_counter':
                    attribute_data = nx.get_node_attributes(self.nx_graph, attribute)
                    attriubte_values = list(attribute_data.values())
                    door_count = [sum(counter.values()) for counter in attriubte_values]
                    df1 = pd.DataFrame({attribute: door_count},
                                       index=list(attribute_data.keys()))
                else:
                    attribute_data = nx.get_node_attributes(self.nx_graph, attribute)
                    df1 = pd.DataFrame({attribute: list(attribute_data.values())},
                                       index=list(attribute_data.keys()))
                df = pd.concat([df, df1], axis=1)

            return df

        else:
            data = []

            for attribute in attributes:
                if attribute == 'door_counter':
                    attribute_data = nx.get_node_attributes(self.nx_graph, attribute)
                    attriubte_values = list(attribute_data.values())
                    door_count = [sum(counter.values()) for counter in attriubte_values]
                    vals = door_count
                else:
                    vals = nx.get_node_attributes(self.nx_graph, attribute).values()
                data.append(np.fromiter(vals, dtype="float"))

            if (len(attributes) > 1):
                np_data = np.stack(data, 1)
            else:
                np_data = data[0]

            keys = np.fromiter(nx.get_node_attributes(self.nx_graph, 'y').keys(),
                               dtype=np.int64)

            return keys, np_data

    def dumpMe(self, path="graphs", name=None):
        """Serializes the object and dumps it through pickle.

        It will automatically create a graphs path to dump the objects.
        The naming convention for the objects is....

        Parameters
        ----------
        path : string
            Specifying the absolute path
        """

        num_trips = len(self.nx_graph.graph['trips'])

        base_trip = self.nx_graph.graph['trips'][0]

        counter = self.nx_graph.graph['it_counter']

        if (name):
            name_string = path + name
        else:
            name_string = ("{}TripID_{}_noTrips_{}"
                           "_graph_iteration_{}.graph").format(path,
                                                               base_trip,
                                                               num_trips,
                                                               counter)

        self.dumpToFile(name_string)

    def dumpToFile(self, full_path):
        # Creates the path if not already existing
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'wb') as file:
            pickle.dump(self, file)

    def exportSubgraph(self, nodes):
        """Based on nodes exports complete subgraph.

        Parameters
        ----------
        nodes : list of ints
            The node ids.

        Returns
        -------
        sub_graph : PoseGraph object (deepcopy)
        """

        sub_graph = PoseGraph()

        reduced_data = self.nx_graph.subgraph(nodes).copy()

        sub_graph.nx_graph = reduced_data

        return sub_graph

    def exportSubgraphNN(self, lat_lon, k=200, d_threshold=1000):
        """Based on the NN nodes to a position generate subgraph.

        Parameters
        ----------
        lat_lon : tuple(float, float)
            Specifies the coordinates of the point to look for the nearest
            neighbors.

        k : int
            The number of NN to use in the view.

        d_treshold : float
            The distance to look for NN.

        Returns
        -------
        sub_graph : PoseGraph object (deepcopy)
        """

        keys, positions = self.getNodeAttributeNumpy(['x', 'y'])

        query_position = self._to_x_y(lat_lon)

        tree = cKDTree(positions)

        idx = tree.query(query_position,
                         k,
                         distance_upper_bound=d_threshold)[1]

        try:
            nodes = list(keys[idx])
        except IndexError:
            print("Not enough a nearest neighbors within threshold!")
            print("Doubling threshold...")
            sub_graph = self.exportSubgraphNN(lat_lon, k, 2 * d_threshold)
            return sub_graph

        sub_graph = self.exportSubgraph(nodes)

        return sub_graph

    def _to_x_y(self, lat_lon):
        """Converts tuple of (lat, lon) to (x, y).

        Parameters
        ----------
        lat_lon : tuple
            Contains lat_lon coordinates.

        Returns
        -------
        x_y : tuple
        """
        x_y = ((lat_lon[1] * self.lat_lon_divide[1]) - self.offset[0],
               (lat_lon[0] * self.lat_lon_divide[0]) - self.offset[1])

        return x_y

    def exportWarping(self):
        """Exports the adjacency list and positions and nose positions.

        The data is then used in the warping algorithm.

        Returns
        -------
        adjacency_list : pandas df
            Index from_id and column to_id.

        node_positions : pandas df
            Index id and x-y-coordinates.
        """

        node_positions = pd.DataFrame(columns=['x', 'y'])

        keys, positions = self.getNodeAttributeNumpy(['x', 'y'])

        node_positions['x'] = positions[:, 0]
        node_positions['y'] = positions[:, 1]

        node_positions.index = keys
        node_positions.index.name = 'id'

        node_positions[['x', 'y']] = node_positions[['x', 'y']].astype('float64',
                                                                       copy=False)

        adjacency_list = list(zip(*list(self.nx_graph.edges)))
        adjacency_df = pd.DataFrame(columns=['from_id', 'to_id'])

        adjacency_df['from_id'] = adjacency_list[0]
        adjacency_df['to_id'] = adjacency_list[1]

        adjacency_df.set_index(['from_id'], inplace=True)

        return node_positions, adjacency_df

    def projectOntoEdge(self, edge_key, query):
        """

        :param query: list of x/y-coordinates and heading of query location
        :param edge_key: key in pose_graph_nx.edge_keys; in case you only know tuple of
                         nodes use pose_graph_nx.getEdgeKey(edge_tuple)
        :return distance: perpendicular distance between query point and line through
                          edge start- and end-node
        :return rel_loc: relative location of query location projected onto edge
        """

        nx_graph = self.nx_graph

        [from_node_id, to_node_id] = self.edge_keys[edge_key]

        from_node = [nx_graph.nodes[from_node_id]['x'],
                     nx_graph.nodes[from_node_id]['y']]

        to_node = [nx_graph.nodes[to_node_id]['x'],
                   nx_graph.nodes[to_node_id]['y']]

        from_node = np.array(from_node)
        to_node = np.array(to_node)
        query = np.array(query)

        dot = (query - from_node).T.dot(to_node - from_node)
        rel_loc = dot / (to_node - from_node).T.dot(to_node - from_node)

        distance = (np.linalg.norm(np.cross(to_node - from_node, from_node - query)) /
                    np.linalg.norm(to_node - from_node))

        return distance, rel_loc

    def updateEdgeKeys(self):
        """
            Give integer value id to edges, and store the nodes representation
            of the networkx DiGraph class in a dict for lookup.
        """
        self.edge_keys = {}
        self.node_keys = {}

        for count, edge in enumerate(self.nx_graph.edges):
            self.edge_keys[count] = edge

    def getEdgeKey(self, edge_tuple):
        return [key for key, value in self.edge_keys.items() if value == edge_tuple][0]

    def getNodeKey(self, node_id):
        return [key for key, value in self.node_ids.items() if value == node_id][0]

    def getNodeHeading(self, nodes_idx=None):
        """Computes the heading of the node_idx specified.

        Node heading is defined as the direction of the edge going out of the
        node, since constant velocity assumption, and hence constant heading,
        holds between consectuive nodes.
        If multiple edges lead out of node, the edges with highest count
        determines the heading.
        If multiple edges with identical counts lead out of node, the edge to
        the lower node id determines the heading.
        If node has no outgoing edge, the heading is determined by the direction
        of the incoming edge.
        If multiple incoming edges exist, the edges with highest count
        determines the heading.
        If multiple incoming edges with identical counts exist, the edge to
        the lower node id determines the heading.

        Parameters
        ----------
        node_idx : list or int

        Returns
        ----------
        headings : list with size like node_idx of heading angle [radians]
        """
        heading = []

        if (nodes_idx is None):
            return []

        nodes_idx = (nodes_idx if type(nodes_idx) is list else [nodes_idx])

        for node_idx in list(nodes_idx):

            successors = list(self.nx_graph.successors(node_idx))
            predecessors = list(self.nx_graph.predecessors(node_idx))
            counts = []

            if len(successors) > 0:
                for successor in successors:
                    counts.append(
                        self.nx_graph.get_edge_data(node_idx,
                                                    successor)['num_vehicles'])

                ref_node_idx = [successor for _, successor in
                                sorted(zip(counts, successors), reverse=True)]

                node_to = ref_node_idx[0]

            elif len(predecessors) > 0:
                # use predecessors instead to define heading
                for predecessor in predecessors:
                    counts.append(
                        self.nx_graph.get_edge_data(predecessor,
                                                    node_idx)['num_vehicles'])

                ref_node_idx = [predecessor for _, predecessor in
                                sorted(zip(counts, predecessors), reverse=True)]

                node_to = node_idx
                node_idx = ref_node_idx[0]

            else:
                # if node has no edges, heading is defined to be zero
                heading.append(0)
                break

            heading.append(math.atan2(self.nx_graph.nodes[node_to]['y'] -
                                      self.nx_graph.nodes[node_idx]['y'],
                                      self.nx_graph.nodes[node_to]['x'] -
                                      self.nx_graph.nodes[node_idx]['x']) + math.pi)

        return heading

    def setAllHeading(self):
        """Sets the heading of all nodes"""

        nodes = self.nx_graph.nodes

        dict_attributes = {}

        for node in nodes:
            heading = self.getNodeHeading(node)
            dict_attributes.update({node: {'heading': heading[0]}})

        nx.set_node_attributes(self.nx_graph, dict_attributes)

    def calculateEdgeDistances(self):
        """ Calculates the 'euclidean' attribute for every edge in the graph,
        by computing the euclidean distance between adjacent nodes.

        Call this whenever the node 'x' or 'y' attributes were changed.
        """
        for u, v in self.nx_graph.edges():
            # update odo_dist, since odo_var only changes
            # when merging graph, i.e. incorporating new measurements
            dist = distance.euclidean([self.nx_graph.nodes[u]['x'],
                                       self.nx_graph.nodes[u]['y']],
                                      [self.nx_graph.nodes[v]['x'],
                                       self.nx_graph.nodes[v]['y']])

            nx.set_edge_attributes(self.nx_graph, {(u, v): {'euclidean_dist': dist}})

    def updateKDTree(self):
        node_ids, poses = self.getNodeAttributeNumpy(['x', 'y'])

        self.node_ids = node_ids

        self.kdtree = cKDTree(poses)
