"""File author: Jens Petit"""

import pandas as pd
from fastdtw import fastdtw
import copy
from scipy.spatial.distance import euclidean
import pdb
from graph_utils import matcher
import scipy
import networkx as nx
from networkx.algorithms.operators.binary import union


def globalXYMerge(graph_base, graph_add):
    """Merges the the information of the additional graph to the base graph.

    This takes place inplace.

    Parameters
    ----------
        graph_base : PoseGraph object
            The graph which is returned in the end and which contains the
            merged information.

        graph_add  : PoseGraph object
            The graph which is added.

    """

    # graph_base.nx_graph = union(graph_add.nx_graph, graph_base.nx_graph)

    unionInPlace(graph_add, graph_base)

    # Adding graph attributes
    graph_base.nx_graph.graph['trips'].extend(graph_add.nx_graph.graph['trips'])
    graph_base.nx_graph.graph['op_dates'].union(graph_add.nx_graph.graph['op_dates'])
    graph_base.nx_graph.graph['lines'].union(graph_add.nx_graph.graph['lines'])


def unionInPlace(graph_add, graph_base):
    """Does an inplace union of two graphs compared to the standard implemented
    union function. This works inplace!

    Parameters
    ----------
    graph_add : PoseGraph object

    graph_base : PoseGraph object
        This object will be manipulated and the graph_add merged into it.
    """

    keys_inter = set(graph_base.nx_graph.nodes).intersection(graph_add.nx_graph.nodes)

    if (not(keys_inter)):
        graph_base.nx_graph.add_nodes_from(graph_add.nx_graph.nodes(data=True))
        graph_base.nx_graph.add_edges_from(graph_add.nx_graph.edges(data=True))
    else:
        raise nx.exception.NetworkXError


def mergeTwoPoseGraphs(graph_base, graph_add, threshold=3):
    """Deprecated, legacy of old merge. Merges two pose graphs as specified in the wiki.

    A new graph object is created based on the graph_base input. The merging of
    two pose nodes takes only place if they are identified as neighbours based
    on Dynamic Time Warping and their distance is below the threshold
    specified.

    Parameters
    ----------
    graph_base : PoseGraph object
        The graph_add object will be merged into this graph. This means all
        nodes of the base graph are kept.

    graph_add : PoseGraph object

    threshold : float
        Distance [m] when two nodes should be merged.

    Returns
    -------
    base_graph : PoseGraph object
        New pose_graph object consisting of the two inputs graphs.

    associations :  pandas df
        Has columns ID_query, ID_graph and distance which correspond to the
        assoassociations drawn during the time warping.
    """

    # Copy object because we want to create new object
    base_graph = copy.deepcopy(graph_base)

    node_base, adj_1 = base_graph.exportWarping()
    node_add, adj_2 = graph_add.exportWarping()

    globalXYMerge(base_graph, graph_add)

    # associations = matcher.Association(graph_add, graph_base)

    # # Iterate over each association and check if they are really close
    # for index, row in associations.iterrows():
    #     if (abs(row['distance']) < threshold):
    #         base_graph.mergeNodes(row['ID_query'], row['ID_graph'])

    associations = 0

    return base_graph, associations
