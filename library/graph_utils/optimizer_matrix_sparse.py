"""File author: Jens Petit"""

import tensorflow as tf
import math
import pdb
import numpy as np
import copy
import os
from graph_utils.db_interface import DBInterface as dbi
from graph_utils import pose_graph_nx as pg
import networkx as nx


class NoDataException(Exception):
    """Costum exception if no data in graph."""
    pass


class NaNGraphException(Exception):
    """Costum exception of nans in graph data present."""
    pass


class Optimizer(object):
    """Optimizing object for pose graphs using matrix formulation of the cost.

    For speed purposes, the original cost function was rephrased and put into a
    matrix format without any outer sums. This makes the computation of the GD
    extremely fast compared with before. The process will just use on GPU which
    is defined on creation of the optimization object.

    Attributes
    ----------
    gpu_options : float (0,1)
        percentage of GPU memory allocated for the process in %
    """

    def __init__(self, gpu_number=0):
        """Constructor which defines that only 50% of GPU memory should be
        used.

        Parameters
        ----------
        gpu_number : int
            Defines which GPU in range 0-3.
        """
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_number)
        self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)

    def toSparseTensor(self, sparse_matrix):
        """Converts a sparse COO matrix to a sparse tensor.

        Parameters
        ----------
        sparse_matrix : scipy COO sparse matrix

        Returns
        -------
        sparse_tensor : tf.SparseTensor
        """

        indices = np.mat([sparse_matrix.row, sparse_matrix.col]).T

        sparse_tensor = tf.SparseTensor(indices, sparse_matrix.data,
                                        sparse_matrix.shape)

        return sparse_tensor

    def optimizeGD(self, pose_graph, max_iter=1000, d_loss_threshold=200,
                   patience=10, eval_every=20, learning_rate=1e-4,
                   plot=None, show_loss=True):
        """Optimize pose graph nodes to better fit constraints.

        Using vanilla gradient descent the original nodes of the pose graph are
        optimized and then this optimized info is stored into a new pose graph
        object. The implementation is based on a matrix formulation which is
        implemented using only sparse tensors.

        Therefore it easily scales to large graphs with more than 100k nodes.

        Parameters
        ----------
        pose_graph : PoseGraph object
            To be optimized.

        max_iter : int
            For the maximum number of iterations.

        d_loss_threshold : float
            Minimum difference of two log every iterations.
            otherwise patience decreased

        patience : int
            Counter for stopping the optimization.

        eval_every : int
            Where iteration mod eval_every == 0 indicates that the loss is
            compared to possible decrease patience.

        learning_rate : float
            Gradient weight to tune convergence and speed.

        plot : Boolean
            if true outputs optimization info

        show_loss : Boolean
            If true outputs loss

        Returns
        -------
        pose_graph_new : PoseGraph object
            The optimized node positions are written to this new object, all
            other values are taken from the old one.
        """

        if (len(pose_graph.nx_graph.nodes) == 0):
            raise NoDataException("There is no data in this graph!")

        # Get data
        keys, positions = pose_graph.getNodeAttributeNumpy(['x', 'y'])

        lambda_odo = 1.0

        x = tf.Variable(positions[:, 0], dtype="float")
        y = tf.Variable(positions[:, 1], dtype="float")

        adj_np = nx.convert_matrix.to_scipy_sparse_matrix(pose_graph.nx_graph,
                                                          format='coo',
                                                          dtype=np.float32)

        adj = self.toSparseTensor(adj_np)

        # Make negative copy to substract sparse
        minus_adj_np = adj_np.copy()
        minus_adj_np.data = -1 * minus_adj_np.data
        minus_adj = self.toSparseTensor(minus_adj_np)

        gps_array = pose_graph.getNodeAttributeNumpy(['gps_x', 'gps_y'])[1]
        gps_positions = tf.constant(gps_array, dtype="float")

        gps_np_cons = pose_graph.getNodeAttributeNumpy(['gps_consistency'])[1]
        gps_consistency = tf.constant(gps_np_cons, dtype="float")

        gps_np_var = pose_graph.getNodeAttributeNumpy(['gps_var_x'])[1]
        gps_variance = tf.constant((1), dtype="float")

        adj_odo_dist = nx.convert_matrix.to_scipy_sparse_matrix(pose_graph.nx_graph,
                                                                weight='odo_dist',
                                                                format='coo',
                                                                dtype=np.float32)

        factor = adj_odo_dist.copy()
        np.reciprocal(1 + factor.data, out=factor.data)
        factor.data = factor.data * lambda_odo
        factor_tf = self.toSparseTensor(factor)
        factor_tf = tf.sparse_reorder(factor_tf)

        adj_odo_dist.data = -1 * (adj_odo_dist.data**2)

        # The squared odometry distance
        odometry_sq = self.toSparseTensor(adj_odo_dist)
        odometry_sq = tf.sparse_reorder(odometry_sq)

        # Odometry loss
        temp_1 = tf.sparse_transpose(tf.sparse_transpose(adj).__mul__(x))
        temp_3 = minus_adj.__mul__(x)
        delta_x = tf.sparse_add(tf.sparse_reorder(temp_3),
                                tf.sparse_reorder(temp_1))

        temp_2 = tf.sparse_transpose(tf.sparse_transpose(adj).__mul__(y))
        temp_4 = minus_adj.__mul__(y)
        delta_y = tf.sparse_add(tf.sparse_reorder(temp_4),
                                tf.sparse_reorder(temp_2))

        distances = tf.sparse_add(tf.square(delta_x), tf.square(delta_y))

        error_adj = tf.square(tf.sparse_add(odometry_sq, distances))

        weighted_errors = tf.multiply(error_adj.values, factor_tf.values)

        error_adj_weighted = tf.SparseTensor(indices=error_adj.indices,
                                             values=weighted_errors,
                                             dense_shape=error_adj.dense_shape)

        odo_loss = tf.sparse_reduce_sum(error_adj_weighted)

        # GPS loss
        gps_error = tf.stack([x, y], 1) - gps_positions
        gps_factor = 5 / gps_variance * tf.square(tf.square(gps_consistency))

        gps_loss = tf.reduce_sum(gps_factor *
                                 tf.reduce_sum(tf.square(gps_error), 1))

        # Total loss and GD init
        total_loss = odo_loss + gps_loss

        train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)
        model = tf.global_variables_initializer()

        losses_eval = []
        patience_current = patience

        with tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_options)) as sess:
            sess.run(model)

            losses_eval.append(total_loss.eval())

            print("Loss before any optimization {}".format(losses_eval[-1]))

            # Iteration loop for GD
            for j in range(max_iter):
                sess.run(train_op)

                if (j % eval_every == 0):
                    current_loss = total_loss.eval()

                    if (math.isnan(current_loss)):
                        raise NaNGraphException("There is some NaN loss!")

                    if (show_loss):
                        print("Current loss {} at iteration {}".format(
                            current_loss, j))

                    # Condition if loss is decreasing and smaller than
                    # threshold
                    if (min(losses_eval) < current_loss or
                        (min(losses_eval) - current_loss < d_loss_threshold)):

                        patience_current = patience_current - 1

                        if (patience_current == 0):
                            print("Patience zero")
                            break

                    else:
                        patience_current = patience
                    losses_eval.append(current_loss)

            if (j == max_iter - 1):
                current_loss = total_loss.eval()
                losses_eval.append(current_loss)
                print("Max iterations run, last loss {}".format(current_loss))

            results_x = dict(zip(keys, x.eval()))
            results_y = dict(zip(keys, y.eval()))

            for key, val in results_x.items():
                val = float(val.item())

            for key, val in results_y.items():
                val = float(val.item())

            print("Nodes in computational graph: " +
                  str(len(sess.graph._nodes_by_name.keys())))

        pose_graph_new = copy.deepcopy(pose_graph)

        nx.set_node_attributes(pose_graph_new.nx_graph, name='x',
                               values=results_x)
        nx.set_node_attributes(pose_graph_new.nx_graph, name='y',
                               values=results_y)

        pose_graph_new.nx_graph.graph['it_counter'] += 1

        pose_graph_new.calculateEdgeDistances()

        pose_graph_new.setAllHeading()

        tf.reset_default_graph()  # necessary to not grow graph too large

        return pose_graph_new


def optimizeDayData(line_no_str, op_date_str, gpu_no=0, to_file=False):
    """Iterates through all the graphs on a single day of a specified line.

    Parameters
    ----------
    line_no_str : tuple of strings
        This is the line number like used from the MVG as a string
        consisting only of numerals.

    op_dat_str : string
        Has the format YYYY-MM-DD.

    gpu_no : int
        On which gpu optimization is run.

    to_file : Boolean
        If true it will write the graph object.

    Returns
    -------
    optimized : list of PoseGraph objects
        Contains all optimized graphs.
    """

    print("Starting to get trip IDs. Takes a while....")
    trip_ids = dbi.getTripsForRouteAndDay(line_no_str, op_date_str)

    pose_graph_list = []

    print("Starting to get graph data... We have {} graphs".format(len(trip_ids)))

    for index, row in trip_ids.iterrows():
        graph = pg.PoseGraph()
        try:
            graph.loadTripFromDb(row['VEH_TRIP_ID'])
            pose_graph_list.append(graph)
        except (TypeError):
            print("NaN data, no interpolation possible")
            pass

    i = 0
    optim = Optimizer(gpu_no)
    optimized = []

    path = "/home/data/graphs/single_optimized_graphs/line_{}/{}"
    path = path.format(line_no_str, op_date_str)

    for graph in pose_graph_list:
        try:
            optimized_graph = optim.optimizeGD(graph)
            optimized.append(optimized_graph)

            if (to_file):
                optimized_graph.dumpMe(path=path)

        except (NoDataException, NaNGraphException):
            print("Hey no data for this graph or nan!")
            pass

        i += 1
        print(i)

    return optimized
