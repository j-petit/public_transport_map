"""File author: Jens Petit"""

import gmaps
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pdb
import math

gmaps.configure(api_key="AIzaSyAxKOxSEB_2NKJhb-1ZQpndQxZ2FKxqZVg")


class Plotter(object):

    """Object represents a plotting object for a pose graph.

    The normal workflow is that the layers are added with possible different
    pose_graphs and then the figure object is called to make the plot.

    Attributes
    ----------
        layers : list of gmaps.layers

        figure : gmaps.figure()

        lat_divide : float
            Number to divide coordinate to get latitude

        lon_divide : float
            Number to diviide coordinate to get longitude

        info_box_template : string
            Template string for filling an info box on gmaps.
    """

    def __init__(self):
        self.layers = []
        self.figure = gmaps.figure()

        self.offset = (861570.226138, 5352505.33286)

        self.lat_divide = 111192.981461485
        self.lon_divide = 74427.2442617538

        self.info_box_template = """
        <dl>
        <dt>Door count: </dt><dd>{0}</dd><dd>{1}</dd>
        </dl>"""

    def plotScatter(self, pose_graph, title='Pose graph'):
        """Creates a simple scatter plot.

        Parameters
        ----------
        pose_graph : PoseGraph object

        title : string
        """

        plt.clf()

        positions = pose_graph.getNodeAttributeNumpy(['x', 'y'])[1]
        plt.scatter(positions[:, 0].flatten(), positions[:, 1].flatten(), s=2)

        plt.title(title)
        plt.draw()
        plt.show()

    def plotGmaps(self):
        """Plots the added layers to gmaps.

        Returns
        -------
        figure : gmaps.figure()
        """

        if not self.layers:
            print("No layers added. Please add before plotting.")

        return self.figure

    def addNodeLayer(self,
                     pose_graph,
                     color=(0, 0, 255, 0.6),
                     nodes_idx=np.empty(0),
                     size=6):
        """Adds node layer based on the nodes.

        Parameters
        ----------
        pose_graph : PoseGraph object

        nodes_idx : list of ints
            Specifies the nodes to be displayed.

        color : (r,g,b,a) tuple
            Defines the color of the nodes.
        """

        color_string_fill = "rgba{}".format(color)

        keys, plot_data = pose_graph.getNodeAttributeNumpy(['x', 'y',
                                                            'gps_consistency'])

        if nodes_idx:
            plot_data = plot_data[[key in nodes_idx for key in keys], :]

        color_strokes = self._consistencyColor(plot_data[:, 2])

        lat_lon = self._latLonTupleList(plot_data[:, 0], plot_data[:, 1],
                                        self.offset)

        pose_layer = gmaps.symbol_layer(lat_lon,
                                        fill_color=color_string_fill,
                                        stroke_color=color_strokes,
                                        scale=size)

        self.figure.add_layer(pose_layer)
        self.layers.append(pose_layer)

    def addNodeLayerFromDataFrame(self, df_nodes, color=(0, 0, 255, 1)):
        """Adds node layer based on the a nodes data frame.

        Parameters:
            df_nodes : nodes pandas df
            color : (r,g,b,a) tuple for the color.
        """

        if (color[3] > 0.1):
            color_stroke = (color[0], color[1],
                            color[2], color[3] - 0.1)
        else:
            color_stroke = (color[0], color[1],
                            color[2], 0)

        coordinates = self._latLonTupleList(df_nodes['x'].as_matrix(),
                                            df_nodes['y'].as_matrix(),
                                            self.offset)

        pose_layer = gmaps.symbol_layer(coordinates,
                                        stroke_color=color_stroke,
                                        scale=4)

        self.figure.add_layer(pose_layer)
        self.layers.append(pose_layer)

    def addEdgeLayerFromXYArrays(self,
                                 start_points,
                                 end_points,
                                 color=(0, 0, 255, 1)):
        """This adds an edge layer based on the nodes and its odometry edges.

        Parameters
        ----------
        start_points : list of locations
            Represented by xy coordinates.

        end_points : list of locations
            Represented by xy coordinates.

        color : tuple(r,g,b,a)
        """

        lines = []

        for start_point, end_point in zip(start_points, end_points):
            x = np.array([start_point[0],
                          end_point[0]])
            y = np.array([start_point[1],
                          end_point[1]])

            coordinates = self._latLonTupleList(x, y, self.offset)

            line = gmaps.Line(start=coordinates[0], end=coordinates[1],
                              stroke_weight=3.0, stroke_color=color)
            lines.append(line)

        edge_layer = gmaps.drawing_layer(features=lines,
                                         show_controls=False, mode='DISABLED')
        self.figure.add_layer(edge_layer)
        self.layers.append(edge_layer)

    def _consistencyColor(self, consistencies):
        """Calculates a shade of black to indicate GPS consistency

        Parameters
        ----------
        consistencies : np.array [1, N]
            With values between 0 and 1.

        Returns
        -------
        color_list : list of rgba tuple
        """

        black_values = consistencies * 255
        black_values = np.floor(black_values)
        black_values = black_values.astype(dtype=int)

        color_list = []

        for i in range(len(black_values)):
            color = (black_values[i], black_values[i], black_values[i], 0.4)

            color_string = "rgba{}".format(color)
            color_list.append(color_string)

        return color_list

    def addEdgeLayer(self, pose_graph, color='gray', time=False, odo=False):
        """This adds an edge layer based on the nodes and its odometry edges.

        Parameters
        ----------
        pose_graph  : pose_graph object

        color : string or tuple(r,g,b,a)

        time : Boolean
            Flag indicating if the line thickness should correspond to the
            time.

        odo : Boolean
            Flag indicating if the line thickness should correspond to the
            difference in euclidean distance and odometry.


        """

        lines = self._addLines(pose_graph, color=color, time=time, odo=odo)
        edge_layer = gmaps.drawing_layer(features=lines,
                                         show_controls=False, mode='DISABLED')
        self.figure.add_layer(edge_layer)
        self.layers.append(edge_layer)

    def _addLines(self, pose_graph, color='gray', time=False, odo=False):
        """Calculates a list of lines for plotting in gmaps between nodes.

        Parameters
        ----------
        pose_graph : PoseGraph object

        color : rgba tuple or specified string

        time : Boolean
            Flag indicating if the line thickness should correspond to the
            time.

        odo : Boolean
            Flag indicating if the line thickness should correspond to the
            difference in euclidean distance and odometry.

        Returns
        -------
        lines : list of gmaps lines
            Lines are defined through their start and ending positions
            according to the corresponding nodes.
        """

        lines = []

        weight = 3

        for edge in pose_graph.nx_graph.edges:

            if (time):
                weight = pose_graph.nx_graph.edges[edge]['travel_time']

            if (odo):
                odo_dist = pose_graph.nx_graph.edges[edge]['odo_dist']
                eucli_dist = pose_graph.nx_graph.edges[edge]['euclidean_dist']

                weight = max(1, 0.5 * np.exp(odo_dist - eucli_dist))

            x = np.array([pose_graph.nx_graph.nodes[edge[0]]['x'],
                          pose_graph.nx_graph.nodes[edge[1]]['x']])
            y = np.array([pose_graph.nx_graph.nodes[edge[0]]['y'],
                          pose_graph.nx_graph.nodes[edge[1]]['y']])

            coordinates = self._latLonTupleList(x, y, pose_graph.offset)

            line = gmaps.Line(start=coordinates[0], end=coordinates[1],
                              stroke_weight=weight, stroke_color=color)
            lines.append(line)

        return lines

    def addHeatmapLayer(self, graphs):
        """Creates a heatmap from multiple graphs

        Parameters
        ----------
            graphs : list of PoseGraph objects

        Returns
        -------
            heatmap : gmaps.heatmap_layer
        """

        list_gps = []

        for graph in graphs:

            plot_data = graph.getNodeAttributeNumpy(['x', 'y'])[1]

            lat_lon = self._latLonTupleList(plot_data[:, 0], plot_data[:, 1],
                                            graph.offset)
            list_gps.extend(lat_lon)

        heatmap = gmaps.heatmap_layer(list_gps)
        self.figure.add_layer(heatmap)
        self.layers.append(heatmap)

        return heatmap

    def addDoorLayer(self, pose_graph, door_hist=False, markers=True):
        """Adds a door layer consisting of gmaps symbols to the figure.

        Parameters
        ----------
        pose_graph : PoseGraph object

        door_hist : Boolean
            Flag indicating if the heatmap of the door events should be drawn
            as visualization of the door distribution.

        markers : Boolean
            Flag indicating if a gmaps marker with information of the door
            event (Counter) should be drawn.
        """

        door_info = []
        x_list = []
        y_list = []

        heatmap_doors = []
        door_heatmap_layer = None

        door_nodes = set()

        for node, door_count in pose_graph.nx_graph.nodes.data('door_counter'):
            if (door_count):
                door_info.append(self.info_box_template.format(door_count, node))
                x_list.append(pose_graph.nx_graph.nodes[node]['x'])
                y_list.append(pose_graph.nx_graph.nodes[node]['y'])
                door_nodes.add(node)

        lat_lon = self._latLonTupleList(np.array(x_list),
                                        np.array(y_list),
                                        pose_graph.offset)

        if (door_hist):

            for node in door_nodes:

                heatmap_doors.extend(pose_graph.nx_graph.nodes[node]['door_hist'])

            x_list = [x[0] for x in heatmap_doors]
            y_list = [x[1] for x in heatmap_doors]

            lat_lon_doors = self._latLonTupleList(np.array(x_list),
                                                  np.array(y_list),
                                                  pose_graph.offset)

            door_heatmap_layer = gmaps.heatmap_layer(lat_lon_doors)
            self.figure.add_layer(door_heatmap_layer)

        door_layer = gmaps.marker_layer(lat_lon, info_box_content=door_info)

        if (markers):
            self.figure.add_layer(door_layer)

        return door_heatmap_layer

    def addAssociationLayer(self, associations, graph_add, graph_base,
                            threshold=3, stroke_weight=3.0,
                            color=(50, 50, 50, 0.4)):
        """Draws lines between the specified associations and the two graphs.

        Parameters
        ----------
        assocations : panda df
            Index is first graph ID1, ID2 is base graph nodes. The distance is
            the euclidean distance of the pair of nodes.

        graph_add : PoseGraph object
            First graph object corresponding to ID_1 (index).

        graph_base : PoseGraph object
            Second graph object correspondig to ID_2.

        threshold : float
            Distance in meters when associations are drawn.

        stroke_weight : float
            Thickness of the lines.

        color : tuple(r,g,b,a)
        """

        lines = []

        for index, row in associations.iterrows():

            if (row['distance'] < threshold):

                id_query = int(row['ID_query'])
                id_base = int(row['ID_graph'])

                x = np.array([graph_add.nx_graph.nodes[id_query]['x'],
                              graph_base.nx_graph.nodes[id_base]['x']])
                y = np.array([graph_add.nx_graph.nodes[id_query]['y'],
                              graph_base.nx_graph.nodes[id_base]['y']])

                coordinates = self._latLonTupleList(x, y, graph_base.offset)

                line = gmaps.Line(start=coordinates[0],
                                  end=coordinates[1],
                                  stroke_weight=stroke_weight,
                                  stroke_color=color)

                lines.append(line)

        assocations_layer = gmaps.drawing_layer(features=lines,
                                                show_controls=False,
                                                mode='DISABLED')

        self.figure.add_layer(assocations_layer)
        self.layers.append(assocations_layer)

    def _latLonTupleList(self, x_coord, y_coord, offset):
        """Returns a list of tuples (lat, lon) for gmaps plotting.

        Parameters
        ----------
        x_coord : numpy array [n]

        y_coord : numpy array [n]

        offset : tuple
            (x, y) offset to calculate (lat, lon)

        Returns
        -------
        lat_lon : list of tuples(lat, lon)
        """
        lat_lon = list(zip((y_coord + offset[1]) / self.lat_divide,
                           (x_coord + offset[0]) / self.lon_divide))

        return lat_lon

    def addHeadingLayer(self, pose_graph, color=(255, 0, 0, 0.6)):
        """Adds a line to each node indicating the heading.

        Parameters
        ----------
        pose_graph : PoseGraph object

        color : tuple(r,g,b,a)
        """

        lines = []

        for node in pose_graph.nx_graph.nodes:

            x_point = pose_graph.nx_graph.nodes[node]['x']
            y_point = pose_graph.nx_graph.nodes[node]['y']

            angle = pose_graph.nx_graph.nodes[node]['heading']

            x_head, y_head = self.translatePoint((x_point, y_point),
                                                 angle=angle,
                                                 distance=10,
                                                 reverse=False)

            x = np.array([x_point, x_head])
            y = np.array([y_point, y_head])

            # returns list of tuples
            coordinates = self._latLonTupleList(x, y, pose_graph.offset)

            # takes tuple of coordinates
            line_head = gmaps.Line(start=coordinates[0], end=coordinates[1],
                                   stroke_weight=5.0, stroke_color=color)

            lines.append(line_head)

        edge_layer = gmaps.drawing_layer(features=lines,
                                         show_controls=False, mode='DISABLED')
        self.figure.add_layer(edge_layer)
        self.layers.append(edge_layer)

    def translatePoint(self, point, angle, distance, reverse=False):
        """Translates a point in the direction of the angle and the distance.

        Parameters
        ----------
        point : tuple(x,y)
            Coordinates of the point to translate.

        angle : float (0,2pi)
            Angle in which direction to translate the point.

        distance : float
            How far should the point be translated.

        Returns
        -------
        new_x : float
            New x coordinate.

        new_y : float
            New y coordinate.
        """
        new_x = point[0] + math.cos(angle) * distance
        new_y = point[1] + math.sin(angle) * distance

        return new_x, new_y
