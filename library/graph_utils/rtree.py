# File author: Christopher Lang

import rtree.index
import pdb
import networkx as nx
import numpy as np

from scipy import math

from graph_utils import pose_graph_nx


class RTreeIndexer(object):
    """
    Maintains an R Tree over a networkx graph with node attributes 'x' and 'y'
    """

    def __init__(self, pose_graph=None):
        self.rtree = None

        if (pose_graph):
            self.build_r_tree(pose_graph)
        else:
            self.pg = None

    def build_r_tree(self, pose_graph, oversize=20):
        """
        Builds an R-Tree indexing all edges in a pose graph and stores the pose graph object.

        :param pose_graph: networkx graph object
        :param oversize: margin of bounding box around each edge
        :return: -
        """
        self.rtree = rtree.index.Rtree()
        self.pg = pose_graph

        nx_graph = pose_graph.nx_graph
        edges_dict = pose_graph.edge_keys

        for edge_key, edge_value in edges_dict.items():
            [from_node_id, to_node_id] = edge_value

            from_node = [nx_graph.nodes[from_node_id]['x'],
                         nx_graph.nodes[from_node_id]['y']]

            to_node = [nx_graph.nodes[to_node_id]['x'],
                       nx_graph.nodes[to_node_id]['y']]

            bounding_box = (min(from_node[0], to_node[0]) - oversize,
                            min(from_node[1], to_node[1]) - oversize,
                            max(from_node[0], to_node[0]) + oversize,
                            max(from_node[1], to_node[1]) + oversize,)

            self.rtree.insert(edge_key, bounding_box)

    def find_closest_edge(self, query, heading=None, radius=20):
        """

        :param query: list of x- and y-coordinates of query location
        :param radius: margin to search box around query location
        :return min_d_edge: key of closest edge
        :return rel_loc: relative location of query location projected onto edge
        """
        
        def getAngleDifference(a1, a2):
            r = (a2 - a1) % (2*np.pi)

            if r >= np.pi:
                r -= 2*np.pi
            return abs(r)

        if self.pg is None:
            print("build_r_tree first!")
            print("exiting...")
            return None, None

        bounding_box = (query[0] - radius,
                        query[1] - radius,
                        query[0] + radius,
                        query[1] + radius,)

        edge_keys = list(self.rtree.intersection(bounding_box, objects=False))

        min_d = 10e5
        rel_loc = None
        min_d_edge = None
        
        nx_graph = self.pg.nx_graph

        for edge_key in edge_keys:

            distance_temp, rel_loc_temp = self.pg.projectOntoEdge(edge_key, query)
            
            if heading is not None:
                # Heading of start node
                (start_node, end_node) = self.pg.edge_keys[edge_key]
                
                edge_heading = math.atan2(nx_graph.nodes[end_node]['y'] 
                                          - nx_graph.nodes[start_node]['y'],
                                          nx_graph.nodes[end_node]['x']
                                          - nx_graph.nodes[start_node]['x']) + math.pi
                
                delta_heading = getAngleDifference(heading, edge_heading)
                
                if delta_heading>(np.pi/4):
                    # query location and edge have different heading
                    # don't match these two
                    continue
                        
            if 0 <= rel_loc_temp <= 1:
                # Point lies in between start and end node.

                if distance_temp < min_d:
                    min_d = distance_temp
                    rel_loc = rel_loc_temp
                    min_d_edge = edge_key

        return min_d_edge, rel_loc, min_d

    def projectOntoEdge(self, edge_key, query):
        """

        :param query: list of x- and y-coordinates of query location
        :param edge_key: key in pose_graph_nx.edge_keys; in case you only know tuple of
                         nodes use pose_graph_nx.getEdgeKey(edge_tuple)
        :return distance: perpendicular distance between query point and line through
                          edge start- and end-node
        :return rel_loc: relative location of query location projected onto edge
        """

        nx_graph = self.pg.nx_graph

        [from_node_id, to_node_id] = self.pg.edge_keys[edge_key]

        from_node = np.array([nx_graph.nodes[from_node_id]['x'],
                              nx_graph.nodes[from_node_id]['y']])

        to_node = np.array([nx_graph.nodes[to_node_id]['x'],
                            nx_graph.nodes[to_node_id]['y']])

        query = np.array(query)

        distance, rel_loc = self.projectOntoLine(from_node, to_node, query)

        return distance, rel_loc

    def projectOntoLine(self, line_start, line_end, query):

        line_start = np.array(line_start)
        line_end = np.array(line_end)
        query = np.array(query)

        dot = (query - line_start).T.dot(line_end - line_start)
        rel_loc = dot / (line_end - line_start).T.dot(line_end - line_start)

        distance = (np.linalg.norm(np.cross(line_end - line_start, line_start - query)) /
                    np.linalg.norm(line_end - line_start))

        return distance, rel_loc
   
