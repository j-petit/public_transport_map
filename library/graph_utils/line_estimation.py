""" File Author: Chengyu Sheu"""
import pickle
import glob
from collections import Counter
from scipy.spatial import cKDTree
from sklearn.preprocessing import LabelEncoder
from numpy.random import shuffle
import numpy as np
from numpy.random import choice
from keras.utils import to_categorical
from keras.layers import Dropout, Activation
from keras.optimizers import RMSprop,SGD
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils import np_utils

def Line(graph, trajectory,row=True,n=3):
    """
    naive approach
    """
    DOOR=DoorFilter(graph)
    if not row:
        position,IDs=convert2row(trajectory)
        door=DoorFilter(trajectory)
        positions=np.empty((0,2))
        for each in range(len(IDs)):
            if IDs[each] in door:
                positions=np.vstack((positions,position[each]))
    else:
        positions=trajectory
    
    
    doors_on_graph=door_match(graph,positions)
    
    counter=Counter()
    for each in doors_on_graph:
        counter += DOOR[each]
    Likelihood=counter
    return counter.most_common(n)

class LineRNN:
    """
    learning approach
    """
    def __init__(self):
        self.training_line = None
        self.training_sample = None
        self.model = None
        self.encoder = None
        pass
        
    def RNNmodel(self):
        self.model = Sequential()
        self.model.add(LSTM(128, input_shape=(self.X_train.shape[1], self.X_train.shape[2]), dropout=0.2, recurrent_dropout=0.2))
        self.model.add(Dense(self.y_train.shape[1]))
        self.model.add(Dropout(0.2)) 
        self.model.add(Activation('softmax'))

        optimizer = RMSprop(lr=0.005)
        #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        
    def fit(self,epochs=10,batch_size=128):
        self.history=self.model.fit(self.X_train, self.y_train,validation_data=(self.X_test, self.y_test),
                          batch_size=batch_size,
                          epochs=epochs)
        
        
    def preprocessing(self,training_x,training_y,DOOR):
        self.look_up=dict(zip(DOOR.keys(),range(len(DOOR))))
        self.look_up[0]=len(DOOR)
        
        
        index=[each for each in range(len(training_y))]
        shuffle(index)

        training_line=[]
        for each in index:
            training_line.append(training_y[each])
        training_sample=[]
        for each in index:
            training_sample.append(training_x[each])

        # truncate and pad input sequences
        max_trajectory_length = 10
        ratio=0.8
        size=len(training_sample)

        X_train = sequence.pad_sequences(training_sample[:int(ratio*size)], maxlen=max_trajectory_length)
        X_test = sequence.pad_sequences(training_sample[int(ratio*size):], maxlen=max_trajectory_length)

        print('Vectorization...')

        X_test=self.OneHot(X_test,max_trajectory_length,DOOR)
        X_train=self.OneHot(X_train,max_trajectory_length,DOOR)

        # encode class values as integers
        self.encoder = LabelEncoder()
        self.encoder.fit(training_line)
        encoded_Y = self.encoder.transform(training_line)

        # convert integers to dummy variables (i.e. one hot encoded)

        #dummy_y = np_utils.to_categorical(encoded_Y)
        from keras.utils import to_categorical
        one_hot=to_categorical(encoded_Y, num_classes=len(set(training_line)))

        y_train= one_hot[:int(ratio*size)]
        y_test=one_hot[int(ratio*size):]
        
        self.X_train=X_train
        self.y_train=y_train
        self.X_test=X_test
        self.y_test=y_test
        
        
    def OneHot(self,training_sample,max_trajectory_length,DOOR):
        x = np.zeros((len(training_sample), max_trajectory_length, len(DOOR)+1), dtype=np.bool)
        for i, trip in enumerate(training_sample):
            for t, stop in enumerate(trip):
                x[i, t, self.look_up[stop]] = 1
        return x
#helper
def random_graph(n=5,path_get = "/home/data/single_optimized_graphs_chengyu/graphs/new/line_*/2018-02-15/*.graph",sl=6,da=1):
    random_trip=glob.glob(path_get)
    print(len(random_trip))
    random_graphs=[]
    for each in choice(random_trip,n):
        temp_graph = pickle.load(open( each, "rb" ))
        temp_line=int(each.split('/')[sl].split('_')[da])
        random_graphs.append((temp_graph,temp_line))
    return random_graphs

def nonrandom_graph(n=5,path_get = "/home/data/single_optimized_graphs_chengyu/graphs/new/line_*/2018-02-15/*.graph"):
    random_trip=glob.glob(path_get)
    random_graphs=[]
    for each in random_trip:
        temp_graph = pickle.load(open( each, "rb" ))
        temp_line=int(each.split('/')[6].split('_')[1])
        random_graphs.append((temp_graph,temp_line))
    return random_graphs

def conti(poses,adj):
    start=(set(adj.index.ravel())-set(adj.values.ravel())).pop()
    position=poses.loc[start]
    IDs=[]
    next_ID=start
    IDs.append(next_ID)
    while next_ID in adj.index:
        next_ID=adj.loc[next_ID].values[0]
        position=np.vstack((position,poses.loc[next_ID].values))
        IDs.append(next_ID)
    return position,IDs

def convert2row(graph):
    poses,adj=graph.exportWarping()
    return conti(poses,adj)

def DoorFilter(graph):
    DOOR={}
    for node, door_count in graph.nx_graph.nodes.data('door_counter'):
        if door_count:
            for each in door_count:
                door_count[each]=1
            DOOR[node]=door_count
    return DOOR

def update_kdtree_door(graph,door):
    poses=np.empty((0,2))
    graph.door_lookup={}
    i=0
    for key in door:
        temp=np.array([graph.nx_graph.nodes[key]['x'],graph.nx_graph.nodes[key]['y']])
        poses=np.vstack((poses,temp))
        graph.door_lookup[i] = key
        i += 1
    graph.door_kdtree = cKDTree(poses)

def door_match(graph,positions):
    door_IDs=[]
    for index in range(positions.shape[0]):
        d, i = graph.door_kdtree.query(positions[index],k=1)
        door_IDs.append(graph.door_lookup[i])
    return door_IDs
