# File author: Christopher Lang

import math
from graph_utils import quad_tree
from itertools import compress

class Indexer(object):
    def __init__(self):
        self.lat_divide = 111192.981461485
        self.lon_divide = 74427.2442617538

    def computeIndexValue(self, x, y, xy_coordinate=True, nodes_idx=[]):
        """
        
        :param nodes_idx: 
        :param x: 
        :param y: 
        :param xy_coordinate: 
        :return: 
        """        
        if xy_coordinate:
            lat, lon = self._latLonConvert(x,y)
        else:
            lat = x
            lon = y

        pre_index = quad_tree.encode(lat, lon)
        pre_index = pre_index[5:]

        i = 0
        new_index = int(pre_index + self._encodeZeros(i))
        while new_index in nodes_idx:
            i = i+1
            new_index = int(pre_index + self._encodeZeros(i)) 

        return new_index


    def findNearestNodes(self, pose_graph, x, y, xy_coordinate=True):
        """
        
        :param pose_graph: 
        :param hash_code: 
        :param radius: 
        :return: 
        """
        
        if xy_coordinate:
            lat, lon = self._latLonConvert(x,y)
        else:
            lat = x
            lon = y
            
        neighborhoods = quad_tree.expand(quad_tree.encode(lat, lon))
        
        nodes_idx = list(pose_graph.nx_graph.nodes)
        nodes_prefix = [node_id.split('_')[0] for node_id in nodes_idx]  
        
        nearest_neighbors = []

        for neighborhood in neighborhoods:

            idx = [prefix==neighborhood for prefix in nodes_prefix]         
            
            if sum(idx)>1:
                nearest_neighbors.extend(compress(nodes_idx, idx))
            elif sum(idx) >0:
                nearest_neighbors.append(compress(nodes_idx, idx))
        
        try:
            # sort results by closeness to query point
            nearest_neighbors.sort(key = lambda nn: (pose_graph.nx_graph.nodes(data='x')[nn] - x)**2 
                                                   +(pose_graph.nx_graph.nodes(data='y')[nn] - y)**2,
                                   reverse=True
                                  )

            return nearest_neighbors
        except:
            return []

    def _latLonConvert(self, x, y, offset=(861570.226138, 5352505.33286)):
        """
        Converts x,y coordinates with to Lat Lon
        :param x:
        :param y:
        :param offset:
        :return:
        """
        return ((y + offset[1]) / self.lat_divide,
                (x + offset[0]) / self.lon_divide)

    def _encodeZeros(self, number):
        """Adds leading zeros to a number if it has less than three digits.

        Parameters
        ----------
        number : int
            Between 1 and 3 digits long int.

        Return
        ------
        number : string
        """

        if (number < 10):
            number = str(number).zfill(2)
        elif (number < 100):
            number = str(number).zfill(1)
        else:
            number = str(number)

        return number
