"""File author: Jens Petit"""

from scipy.spatial import cKDTree
from scipy.spatial.distance import euclidean
from scipy.spatial import Delaunay
import networkx as nx
import numpy as np
import copy
from graph_utils import merger
import math
import pdb
import pandas as pd
from collections import Counter
from networkx.algorithms.traversal import depth_first_search as dfs
from networkx.algorithms.shortest_paths.weighted import dijkstra_path


class Representor(object):
    """The object creates meaningful map out of the raw data.


    Attributes
    ----------
    adj : scipy sparse csr matrix
        Adjacency matrix of the graph.

    repr_dict : dictionary
        A dictionary of sets which has the centroid as key and then as set all
        nodes which are assigned to the centroid.

    keys : numpy array
        All keys of the to be represented pose graph.

    n : int
        The number of nodes in the graph.

    point_assigned : dictionary
        Contains as a key nodes and then the assigned centroid. Its the other
        way around than the repr_dict.

    all_points : set of ints
        contains all points with ids from 0 to n where n is the number of
        nodes.

    positions : numpy array (n, 2)
        x,y coordinates of each node.

    headings : numpy array (n)
        Heading information of each node.

    tree_pos : KDTree
        Nearest neighbor search tree with all nodes positions.

    centroid_means : numpy array (r, 2)
        All the mean position of the centroids

    intermediate_graph : PoseGraph object
        The intermediate new graph with all edges drawn to original vehicle
        data. This is before the edge sparsification with the search is
        applied.

    centroids_info : dict() of tuple(mean, heading)

    doors : numpy array
        Contains the door counter.

    delaunay : Delaunay()
        Delaunay Triangulation of all nodes.

    key_trans : dict()
        Translator dictionary from ordered numpy row to node ids.
    """

    def __init__(self, pose_graph):
        """Constructor with a pose graph object.

        Parameters
        ----------
        pose_graph : PoseGraph object
        """

        pose_graph.setAllHeading()

        self.adj = nx.convert_matrix.to_scipy_sparse_matrix(pose_graph.nx_graph,
                                                            format='csr')

        # dict of sets with reprsentative and all assigned GPS points
        self.repr_dict = dict()

        self.point_assigned = dict()

        self.keys, data = pose_graph.getNodeAttributeNumpy(['x', 'y',
                                                            'heading',
                                                            'door_counter'])

        self.all_points = set(range(len(self.keys)))

        self.positions = data[:, 0:2]

        self.centroid_edges = {}

        self.headings = data[:, 2]

        self.doors = data[:, 3]

        self.tree_pos = cKDTree(self.positions)

        self.n = len(self.keys)

        self.pose_graph = pose_graph

        self.centroid_means = None

        self.centroids_info = {}

        self.delaunay = Delaunay(self.positions, incremental=False)

        self.key_trans = dict(zip(range(self.n), self.keys))

    def assignNode(self, represent_id, point_id, represented=False):
        """Assigns a node to a centroid.

        If an assignment already exists overwrite it and delete the point out
        of the set of nodes to the old centroid.

        Parameters
        ----------
        represent_id : int
            The centroids id.

        point_id : int
            The nodes id.

        represented : Boolean
            Flag which indicates if the point is already represented through a
            centroid.
        """

        if (represented):
            if (self.point_assigned[point_id] != represent_id):
                self.repr_dict[self.point_assigned[point_id]].remove(point_id)

        self.repr_dict[represent_id].update({point_id})
        self.point_assigned[point_id] = represent_id

    def getCentroidData(self, representor):
        """Computes mean position and heading of a centroid.

        Parameters
        ----------
        representor : int
            The centroid for which the data should be calculated.

        Returns
        -------
        mean : numpy array (2)
            Contains the mean position.

        heading : float
            Contains the median heading in the centroid.
        """

        mean = np.mean(self.positions[list(self.repr_dict[representor])],
                       axis=0)

        heading = np.median(self.headings[list(self.repr_dict[representor])])

        return mean, heading

    def getNextNodeDelaunay(self, centroid_points):
        """The next neighboring node is chosen which is not yet represented.

        Neighboring is defined by the delaunay triangulation of all points.

        Parameters
        ----------
        centroid_points : set of points
            All points to look for neighbors.

        Returns
        -------
        point : int
            Neighboring point which is not yet represented.
        """

        def find_neighbors(pindex, triang):
            """Returns neighbors in delaunay triangluation"""

            neighbors = (triang.vertex_neighbor_vertices[1]
                         [triang.vertex_neighbor_vertices[0]
                          [pindex]:triang.vertex_neighbor_vertices[0][pindex + 1]])

            return neighbors

        for point in centroid_points:

            candidates = find_neighbors(point, self.delaunay)

            # candidates = self.delaunay.simplices[point, :]

            for candidate in candidates:
                if (not(candidate) in self.point_assigned.keys()):
                    return candidate

        not_assigned = self.all_points - set(self.point_assigned.keys())

        return not_assigned.pop()

    def computeCentroids(self, radius=30, angle=1, cutoff_percent=10):
        """Following two step clustering appraoch to get centroids of spatial
        data.

        1) Pick a random node. Compute neighbors in a radius and similar angle.
        2) Place centroid at mean position of neighbors.
        3) Compute neighbors of centroid in a radius and similar angle.
        4) Replace centroid at mean of these neighbors.
        5) Pick a new node outside of the centroid.
        6) Until all nodes are assigned a centroid.

        Parameters
        ----------
        radius : float
            Specifying the radius in which neighbor search is performed.

        angle : float
            Deviation in angle which is allowed for neighbors.

        cutoff_percent : int (0,100)
            Centroids which cumulative represent less nodes than the percentage
            or dropped.
        """

        # Initial with random point
        current = np.random.randint(0, len(self.keys))

        # Loop until all points are assigned
        while (len(self.point_assigned.keys()) != len(self.keys)):

            number_assigned = len(self.point_assigned.keys())

            if (number_assigned % 100 == 0):
                print("{} of {} are represented".format(number_assigned,
                                                        len(self.all_points)))

            current_pos = self.positions[current, :]
            current_heading = self.headings[current]
            current_door = bool(self.doors[current])

            current_centroid = len(self.repr_dict)
            self.repr_dict.update({current_centroid: {current}})
            self.point_assigned.update({current: current_centroid})

            neighborhood = self.tree_pos.query_ball_point(current_pos, r=radius)
            neighborhood = set(neighborhood)
            neighborhood.discard(current)
            not_represented = neighborhood - set(self.point_assigned.keys())

            temp_nodes = set()
            temp_nodes.add(current)

            for unrepresented in not_represented:

                test_door = bool(self.doors[unrepresented])

                if (self.areNeighbors(self.headings[unrepresented],
                                      current_heading, angle, current_door,
                                      test_door)):

                    temp_nodes.add(unrepresented)

            mean = np.mean(self.positions[list(temp_nodes)], axis=0)
            heading = np.median(self.headings[list(temp_nodes)])

            near_centroid = self.tree_pos.query_ball_point(mean, r=radius)
            near_centroid = set(near_centroid)

            near_centroid_unrep = near_centroid - set(self.point_assigned.keys())
            near_centroid_repr = near_centroid - near_centroid_unrep

            for unrepresented in near_centroid_unrep:
                test_door = bool(self.doors[unrepresented])
                if (self.areNeighbors(self.headings[unrepresented],
                                      heading, angle, current_door,
                                      test_door)):

                    self.assignNode(current_centroid, unrepresented)

            for represented in near_centroid_repr:
                test_door = bool(self.doors[represented])
                if (self.areNeighbors(self.headings[represented],
                                      heading, angle, current_door,
                                      test_door)):

                    distance_new_centroid = euclidean(self.positions[represented],
                                                      mean)

                    mean_assigned_centroid = self.getCentroidData(self.point_assigned[represented])[0]

                    distance_assign_centroid = euclidean(self.positions[represented],
                                                         mean_assigned_centroid)

                    if (distance_new_centroid < distance_assign_centroid):

                        self.assignNode(current_centroid,
                                        represented,
                                        represented=True)

            try:
                current = self.getNextNodeDelaunay(self.repr_dict[current_centroid])
            except KeyError:
                print("No more unassigned nodes... Done here!")

        cutoff = self.getCutoffForCentroids(cutoff_percent)

        self.filterCentroids(cutoff)

        self.setCentroidsInfo()

    def getCutoffForCentroids(self, cutoff_percent):
        """Computes the cutoff based on cumulative representation of nodes.

        The centroids explaining less than cutoff percentage of nodes should be
        dropped. The exact number of nodes which corresponds to this is
        calculated here through a cumulative histogramm approach.

        Parameter
        ---------
        cutoff_percent : int (0,100)

        Returns
        -------
        cutoff : int
            Centroids with less than cutoff nodes are dropped.
        """

        size_hist_list = []

        for centroid, nodes in self.repr_dict.items():
            size_hist_list.append(len(nodes))

        cutoff = self._getCutoffGeneral(size_hist_list, cutoff_percent)

        print("The node cutoff is {} representing less than {}%".format(cutoff,
                                                                        cutoff_percent))

        return cutoff

    def getCutoffForEdges(self, cutoff_percent):
        """Computes the cutoff based on cumulative representation of edges.

        The edges explaining less than cutoff percentage of motions should be
        dropped. The exact number of vehicles which corresponds to this is
        calculated here through a cumulative histogramm approach.

        Parameter
        ---------
        cutoff_percent : int (0,100)

        Returns
        -------
        cutoff : int
            Edges with less than cutoff vehicles on them are dropped.
        """

        counters = nx.get_edge_attributes(self.new_graph.nx_graph, 'Counter')
        size_hist_list = list(counters.values())

        cutoff = self._getCutoffGeneral(size_hist_list, cutoff_percent)

        print("The edge cutoff is {} representing less than {}%".format(cutoff,
                                                                        cutoff_percent))

        return cutoff

    def exportCentroidsInGraph(self,
                               p_cutoff_edge_final=5,
                               p_cutoff_edge_inter=2):

        """Exports the graph with the current centroids as only nodes.

        This is the main function to execute after computeCentroids(). It
        creates a new graph which is then filled with the representatives and
        its attributes (nodes only so far). Then the new edges are computed in
        containing information in a two step procedure.

        1) Connect the nodes through the vehicle movements. Create the edges
        which connect a pair of centroids which has been visited by a vehicle
        trajectory. Additionally, connect centroids which are close and similar
        in heading. This gives the intermediate graph.

        2) Now the intermediate graph is sparsified. The vehicle trajectories
        are again overlayed and then a search is performed with the modified
        euclidean distance as a weight. Shortest path is found over which then
        the vehicle information is split.

        Parameters
        ----------
        p_cutoff_edge_inter : int(0, 100)
            The cutoff of unrepresentative edges for the intermediate graph.

        p_cutoff_edge_final : int(0, 100)
            The cutoff of unrepresentative edges for the new centroid graph.

        Returns
        -------
        new_graph : PoseGraph object
            The map like final new graph.
        """

        setter_dict = {}

        centr_trans = {}

        # Export only new centroid center nodes
        for centroid, representents in self.repr_dict.items():

            mean, heading = self.getCentroidData(centroid)

            representor = representents.pop()
            node_graph_id = self.keys[representor]
            representents.add(representor)

            projected_doors = None

            if (self.pose_graph.nx_graph.nodes[node_graph_id]['door_counter']):
                door_counter = self.unionizeCounters(representents)

                projected_doors = self.doorHistogram(representents, mean,
                                                     heading)

                # Add points as Histogramm. Project each
            else:
                door_counter = Counter()

            centr_trans.update({centroid: node_graph_id})

            setter_dict.update({node_graph_id: {'x': mean[0], 'y': mean[1],
                                                'heading': heading,
                                                'door_counter': door_counter,
                                                'door_hist': projected_doors}})

        # Create new graph (first intermediate) with only centroids and remove
        # all edges
        self.new_graph = self.pose_graph.exportSubgraph(list(setter_dict.keys()))
        nx.set_node_attributes(self.new_graph.nx_graph, setter_dict)
        all_edges = copy.deepcopy(self.new_graph.nx_graph.edges)
        self.new_graph.nx_graph.remove_edges_from(all_edges)

        # Add intermediate edges
        new_edges = self.calcCentroidEdgesTrace(trans_dict=centr_trans,
                                                cutoff_percent=p_cutoff_edge_inter)

        # Remove nodes without any edges
        isolated_nodes = list(nx.isolates(self.new_graph.nx_graph))
        self.new_graph.nx_graph.remove_nodes_from(isolated_nodes)

        # Remove all intermediate edges
        old_edges = copy.deepcopy(self.new_graph.nx_graph.edges)
        self.new_graph.nx_graph.remove_edges_from(old_edges)

        # Add new final edges
        self.averageEdges(new_edges)
        self.new_graph.nx_graph.add_edges_from(new_edges)
        nx.set_edge_attributes(self.new_graph.nx_graph, new_edges)

        # Remove not representative edges
        cutoff_edges = self.getCutoffForEdges(p_cutoff_edge_final)
        self._removeEdgesOnCutoff(cutoff_edges)

        self.new_graph.calculateEdgeDistances()

        return self.new_graph

    def filterCentroids(self, cutoff):
        """Only keeps centroids which contain more nodes then the cutoff.

        Parameters
        ----------
        cutoff : int
            Threshold is set of nodes of threshold is less than, then it is
            dropped.
        """

        remove_list = []

        for centroid, representents in self.repr_dict.items():
            if (len(representents) < cutoff):
                remove_list.append(centroid)

        for centroid in remove_list:

            for node in self.repr_dict[centroid]:
                del self.point_assigned[node]

            del self.repr_dict[centroid]

    def consistencyCheck(self):
        """Checks if both the main dictionaries are consistent in their
        representation"""

        for key, centroid in self.point_assigned:

            if not(key in self.repr_dict[centroid]):
                return False

        return True

    def setCentroidsInfo(self):
        """Sets the centroids mean position and heading."""

        num_centroids = len(self.repr_dict.keys())

        self.centroid_means = np.zeros((num_centroids, 2))

        for i, (centroid, nodes) in enumerate(self.repr_dict.items()):

            mean, heading = self.getCentroidData(centroid)

            self.centroid_means[i, :] = mean
            self.centroids_info.update({centroid: (mean, heading)})

            self.headings[i] = heading

    def calcCentroidEdgesTrace(self, trans_dict, cutoff_percent=2):
        """Iterates through all nodes and traces the trajectories of the
        vehicles on the centroids.

        If a connection on the centroids is found then an edge is added. The
        least expressive edges are removed in a final step.

        Parameters
        ----------
        trans_dict : dict
            The key is a centroid from 0 to number of centroids and the value
            is the chosen pose graph node id which represents the centroid.

        cutoff_percent : int(0,100)
            The least percentage of least expressive edges to drop.
        """

        centroid_start = np.nan
        centroid_end = np.nan
        point_start = np.nan
        point_end = np.nan

        in_search = False

        # List to give to the final edge calculation with the snapping
        transform_list = []

        for i in range(self.n):

            if not(i in self.point_assigned.keys()):
                if (self.adj[i, :].nnz == 0):
                    in_search = False
                continue

            # Found a centroid connection
            if (in_search and self.point_assigned[i] != centroid_start):

                in_search = False
                centroid_end = self.point_assigned[i]
                point_end = i

                edge_feasible = self.edgeFeasibility(centroid_start,
                                                     centroid_end)

                if edge_feasible:

                    edge = (trans_dict[centroid_start],
                            trans_dict[centroid_end])

                    if edge in self.new_graph.nx_graph.edges:
                        attributes = {'Counter': 1 +
                                      self.new_graph.nx_graph.edges[edge]['Counter']}
                    else:
                        attributes = {'Counter': 1}

                    self.new_graph.nx_graph.add_edge(edge[0], edge[1], **attributes)
                    transform_list.append((edge, (point_start, point_end)))

            # Trajectory ends -> seach ends
            if (self.adj[i, :].nnz == 0):
                in_search = False
            elif (not(in_search)):
                centroid_start = self.point_assigned[i]
                point_start = i
                in_search = True

        # Remove intermediate edges which are not representative
        edge_cutoff = self.getCutoffForEdges(cutoff_percent)
        self._removeEdgesOnCutoff(edge_cutoff)

        # Second edge finding algorithm based on the heading an region in front
        # of each centroid.
        self.calcCentroidEdgesRegion(radius=30, trans_dict=trans_dict)
        self.intermediate_graph = copy.deepcopy(self.new_graph)

        final_edges = self.snapTrajectories(transform_list)

        return final_edges

    def edgeFeasibility(self, centroid_start, centroid_end,
                        distance=150, angle=2):
        """Checks if an edge is feasible depending on distance and angle.

        Parameters
        ----------
        centroid_start : int
            Centroid ID where the edge start.

        centroid_end : int
            Centroid ID where the edge ends.

        distance : float
            Maximum distance an edge can connect.

        angle : float (0, 2pi)
            Maximum heading angle difference between two centroids.
        """

        start_info = self.centroids_info[centroid_start]
        end_info = self.centroids_info[centroid_end]

        if (euclidean(start_info[0], end_info[0]) < distance and
                self.deltaAngle(start_info[1], end_info[1]) < angle):

            return True

        return False

    def unionDictOfLists(self, dict_1, dict_2):
        """Unionizes two dicts where the values are special. INPLACE!

        The first one is a counter which should be added together. The other
        ones are lists which should be extended.

        Parameters
        ----------
        dict_1 : dict

        dict_2 : dict

        Returns
        -------
        dict_1 : dict
        """

        dict_1['Counter'] += dict_2['Counter']
        dict_1['travel_time'].extend(dict_2['travel_time'])
        dict_1['odo_dist'].extend(dict_2['odo_dist'])

        return dict_1

    def translatePoint(self, point, angle, distance):
        """Translates a point in the direction of the angle and the distance.

        Parameters
        ----------
        point : tuple (float, float)
            Point to translate.

        angle : float (0, 2pi)
            Polar coordinate angle to translate point.

        distance : float
            How far from the point out should the new point be.

        Returns
        -------
        new_point : tuple(float, float)
            The translated point.
        """

        new_x = point[0] + math.cos(angle) * distance
        new_y = point[1] + math.sin(angle) * distance

        new_point = np.array([new_x, new_y])

        return new_point

    def deltaAngle(self, alpha_1, alpha_2):
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

    def snapTrajectories(self, transform_list, cutoff_percent=5, depth_limit=5,
                         power_dijkstra=1.5):
        """Snaps the trajectories to the centroids using dijkstra search

        A vehicle trajectory is attached to an order of centroids. Now the
        trajectory of the vehicle is recalculated based on the intermediate
        graph and a search between two adjacent (trajectory vise) centroids.
        The weight is a powered euclidean distance.

        Parameters
        ----------
        transform list : tuple(tuple(), tuple())
            Contains the trajectory split in movements between centroids and
            start and end points in the original trajectories. The first tuple
            is the edge connecting two centroids, the second element of the
            tuple is the start and end point of the original trajectory ids.

        cutoff_percent : int (0, 100)
            Unrepresentative edges to drop.

        depth_limit : int(0,10)
            In which maximum depth to look for the second centroid on the
            intermediate graph.

        power_dijkstra : float (1, 5)
            The exponent of the powered euclidean distance used as a weight in
            the dijkstra search algorithm.

        Returns
        -------
        final_edges : dict of dict (attributes)
            The final edges with the counter, time and odometry information.
        """

        nx.set_edge_attributes(self.new_graph.nx_graph,
                               values=0,
                               name='dijkstra_dist')

        final_edges = {}

        self.calculateEdgeDistances(power=power_dijkstra)

        for edge, point in transform_list:

            nodes_reachable = dfs.dfs_preorder_nodes(self.new_graph.nx_graph,
                                                     source=edge[0],
                                                     depth_limit=depth_limit)

            is_reachable = False

            for node in nodes_reachable:
                if node == edge[1]:
                    is_reachable = True
                    break

            if (is_reachable):
                path = dijkstra_path(self.new_graph.nx_graph, source=edge[0],
                                     target=edge[1], weight='dijkstra_dist')

                num_edges = len(path) - 1

                temp_edges = [(path[i], path[i + 1])
                              for i in range(num_edges)]

                split_info = self.pointSplitEdgeAttributes(point, num_edges)

                for edge in temp_edges:

                    attributes = {'Counter': 1, 'odo_dist': [split_info[0]],
                                  'travel_time': [split_info[1]]}

                    if edge in final_edges.keys():
                        attributes = self.unionDictOfLists(attributes,
                                                           final_edges[edge])

                    final_edges.update({edge: attributes})

        return final_edges

    def pointSplitEdgeAttributes(self, path, num_splits):
        """Splits a path of the original vehicle into multiple edges

        Parameters
        ----------
        path : tuple(start_point, end_point)
            Which nodes this path contains.

        num_splits : int
            Into how many edges this should be split uniformely.

        Returns
        -------
        new_info : tuple(distance, travel_time)
        """

        distance = 0
        travel_time = 0

        temp_edges = [(self.key_trans[i], self.key_trans[i + 1])
                      for i in range(path[0], path[1])]

        for edge in temp_edges:
            distance += self.pose_graph.nx_graph.edges[edge]['odo_dist']
            travel_time += self.pose_graph.nx_graph.edges[edge]['travel_time']

        new_info = (distance / num_splits, travel_time / num_splits)

        return new_info

    def calculateEdgeDistances(self, power=1.5):
        """ Calculates the 'dijkstra_dist' attribute for every edge in the
        graph, by computing the euclidean distance and taking it to a power
        between adjacent nodes.

        Call this before snapping trajectories the node 'x' or 'y' attributes
        were changed. The information is stored in the new graph as
        dijkstra_dist attribute.

        Parameters
        ----------
        power : float
            The exponent the euclidean distance is taken to.
        """

        for u, v in self.new_graph.nx_graph.edges():

            distance = euclidean([self.new_graph.nx_graph.nodes[u]['x'],
                                  self.new_graph.nx_graph.nodes[u]['y']],
                                 [self.new_graph.nx_graph.nodes[v]['x'],
                                  self.new_graph.nx_graph.nodes[v]['y']])**power

            self.new_graph.nx_graph.edges[u, v]['dijkstra_dist'] = distance

    def calcCentroidEdgesRegion(self, radius, trans_dict):
        """Makes an edge between centroids which follow each other and have
        similar heading.

        The search region is a circle in front of the centroid where in front
        means in the heading direction. The edges are added to the graph.

        Parameters
        ----------
        radius : float
            Distance in which to look for other centroids.

        trans_dict : dict()
            Contains the translation of centroid ids to original pose graph
            node ids which are used to represent the graph.

        Returns
        -------
        edges: list of tuples(int, int)
        """

        centroid_tree = cKDTree(self.centroid_means)

        assignments = list(self.repr_dict.keys())

        for i, (centroid, nodes) in enumerate(self.repr_dict.items()):

            search_point = self.translatePoint(self.centroid_means[i, :],
                                               self.headings[i],
                                               distance=30)

            neighbors = centroid_tree.query_ball_point(search_point,
                                                       r=radius)

            neighbors = set(neighbors)

            try:
                neighbors.remove(i)
            except KeyError:
                pass

            neighbors_keys = [assignments[x] for x in neighbors]

            for neighbor in neighbors_keys:

                neighbor_heading = self.getCentroidData(neighbor)[1]

                if (self.deltaAngle(self.headings[i], neighbor_heading) < 1.6):

                    edge = (trans_dict[centroid],
                            trans_dict[neighbor])

                    if edge in self.new_graph.nx_graph.edges:
                        attributes = {'Counter': 1 +
                                      self.new_graph.nx_graph.edges[edge]['Counter']}
                    else:
                        attributes = {'Counter': 1}

                    self.new_graph.nx_graph.add_edge(edge[0], edge[1], **attributes)

    def averageEdges(self, edges):
        """Averages the edge information of time and distance and computes
        their respective variance. INPLACE!

        Parameter
        ---------
        edges : dict of dict
            Contains the edge information in the form {edge: attributes}
        """

        for edge in edges:
            edges[edge].update({'odo_var': np.std(edges[edge]['odo_dist'])})
            edges[edge].update({'odo_dist': np.median(edges[edge]['odo_dist'])})
            edges[edge].update({'time_var': np.std(edges[edge]['travel_time'])})
            edges[edge].update({'travel_time': np.median(edges[edge]['travel_time'])})

    def _getCutoffGeneral(self, hist_list, cutoff_percent):
        """Computes the threshold of the least representives entities.

        A histogram gives the occurance of counters. For example the list [1,
        1, 3, 4, 5] implies that there are two times objects of size one, and
        one time objects of sizes 3, 4 and 5.

        Parameters
        ----------
        hist_list : list
            Contains the size counter objects

        cutoff_percent : int (0,100)
            The least cutoff_percent representantives will under the
            cutoff_threshold.

        Returns
        -------
        cutoff_treshold : int
            Size from which then on the entities represent more than the least
            expressive cutoff_percent objects.
        """

        hist_counts = np.bincount(np.array(hist_list))

        multiply = np.arange(0, len(hist_counts))

        final_count = np.multiply(hist_counts, multiply)

        cum_hist = np.cumsum(final_count) / np.sum(final_count)

        cutoff_threshold = np.argmax(cum_hist > cutoff_percent / 100) - 1

        return cutoff_threshold

    def _removeEdgesOnCutoff(self, edge_cutoff):
        """Removes the edges which have counter lower than cutoff.

        Additionally, it is checked whether the edge has connects nodes with
        only one out or in edge. If this is the case it is not removed because
        it might destroy the overall trajectory.

        Parameters
        ----------
        edge_cutoff : int (0,n)
            Number if counter small the edge is removed.
        """

        it_list = list(self.new_graph.nx_graph.edges.data('Counter'))

        for (u, v, counter) in it_list:
            if (counter < edge_cutoff and
                    self.new_graph.nx_graph.out_degree(u) > 1 and
                    self.new_graph.nx_graph.in_degree(v) > 1):
                self.new_graph.nx_graph.remove_edge(u, v)

    def getNextNode(self, centroid_points):
        """Gets a neighboring node which is not yet represented.

        Neighboring is defined through the del

        Parameters
        ----------
        centroid_points : set of points

        Returns
        -------
        neighbor_point : int
            A point which is neighbor to any point in the centroid. If no
            neighbor is available then it is a random unrepresented point.
        """

        for point in centroid_points:

            if (self.adj[point, :].nnz != 0):
                neighbor_point = point + 1

                if (not(neighbor_point) in self.point_assigned.keys()):
                    return neighbor_point

        not_assigned = self.all_points - set(self.point_assigned.keys())

        return not_assigned.pop()

    def areNeighbors(self, angle_1, angle_2, d_angle, is_door_1, is_door_2):
        """Checks if two nodes are neighbors based on their angle and if they
        are door node or not.

        Parameters
        ----------
        angle_1 : float
            Heading angle of node 1.

        angle_2 : float
            Heading angle of node 2.

        d_angle : float
            Delta angle which is allowed to be considered a neighbor.

        is_door_1 : boolean
            Flag which indicates whether node 1 is a door node.

        is_door_2 : boolean
            Flag which indicates whether node 2 is a door node.
        """

        door_match = (is_door_1 == is_door_2)

        if (door_match):
            delta_alpha = self.deltaAngle(angle_1, angle_2)

            if (delta_alpha < d_angle):
                return True

        return False

    def unionizeCounters(self, node_ids):
        """Unionizes the door Counter() objects of many door nodes.

        Parameters
        ----------
        node_ids : iterable of node ids

        Returns
        -------
        door_counts : Counter()
            Union of individual Counter()
        """

        door_counts = Counter()

        for node_id in node_ids:
            node_id_pg = self.keys[node_id]
            door_counts.update(self.pose_graph.nx_graph.nodes[node_id_pg]['door_counter'])

        return door_counts

    def doorHistogram(self, representants, mean, heading):
        """Projects all original gps data to the door line.

        Parameters
        ----------
        representants : set()
            Node Ids in numpy space.

        mean : np.array [2]
            Mean of the door centroid.

        heading : float
            Heading of the centroid.

        Returns
        -------
        projected_points : list of tuples(x,y)
        """

        end_line = self.translatePoint(mean, heading, 1)

        projected_points = []

        for represent in representants:

            node_id = self.key_trans[represent]

            query = (self.pose_graph.nx_graph.nodes[node_id]['gps_x'],
                     self.pose_graph.nx_graph.nodes[node_id]['gps_y'])

            point = self.projectOntoLine(mean, end_line, query)

            projected_points.append(point)

        return projected_points

    def projectOntoLine(self, line_start, line_end, query):
        """Projects a query point onto a line and gives the new coordinates.

        Parameters
        ----------
        line_start : numpy array [2]
            Start coordinates of the line x, y.

        line_end : numpy array [2]
            End coordinates of the line x, y.

        query : numpy array [2]
            Point to be projected onto the line.

        Returns
        -------
        new_point : numpy array [2]
            Projected coordinates of the query point.
        """

        line_start = np.array(line_start)
        line_end = np.array(line_end)
        query = np.array(query)

        line = line_end - line_start

        dot = (query - line_start).T.dot(line)
        rel_loc = dot / (line).T.dot(line)

        rel_translation = (query - line_start) - rel_loc * (line)

        new_point = query - rel_translation

        return new_point
