"""Author: Chengyu Sheu, Georg Aures"""
import pandas as pd
import pdb
import numpy as np
from scipy.spatial import cKDTree
from numpy.linalg import norm
from numpy import inner
from collections import deque

def heading(query,nodes,adj,adj_inverse):
    """
    point heading
    """
    if query in adj.index:
        forward=adj.loc[query].values
    else:
        forward=np.empty((0,2))
    if query in adj_inverse.index:
        backward=adj_inverse.loc[query].values
    else:
        backward=np.empty((0,2))
    heading=[]
    i=0
    for each_f in forward:
        for each_b in backward:
            if each_f!=each_b:
                heading.append(nodes.loc[each_f].values-nodes.loc[each_b].values)
                heading[-1]/=norm(heading[-1])
        i+=1
    if i==0:
        ###error
        if forward.shape[0]!=0:
            heading.append(nodes.loc[forward[0]].values-nodes.loc[query].values)
        elif backward.shape[0]!=0:
            heading.append(nodes.loc[query].values-nodes.loc[backward[0]].values)
    return np.array(heading)

def find_z_heading(d,x_heading,z_list,nodes_z,adj,adj_inverse,threshold=0.5):
    i=0
    inner_best=[]
    for each in z_list:
        z_heading=heading(each,nodes_z,adj,adj_inverse)
        inner_xz=[]
        for each_z in z_heading:
            for each_x in x_heading:
                inner_xz.append(inner(each_x,each_z))
        if len(inner_xz)>0:
            inner_best.append(np.nanmax(inner_xz))
        else:
            inner_best.append(np.nan)
    inner_best=np.array(inner_best)
    if len(inner_best)==0:
        return np.nan
    if np.nanmax(inner_best)>=threshold:
        return z_list[inner_best>=threshold][np.nanargmin(d[inner_best>=threshold])]
    else:
        return np.nan

def Association_heading(query,graph,r,threshold=0.5,anti_loop=True):
    nodes1,adj1=query.exportWarping()
    nodes2,adj2=graph.exportWarping()
    
    len_query=nodes1.shape[0]
    adj1_inverse=adj1.reset_index().set_index('to_id')
    adj2_inverse=adj2.reset_index().set_index('to_id')
    tree = cKDTree(nodes2)

    association_temp=[]
    for i in range(len_query):
        
        x_heading=heading(nodes1.index[i],nodes1,adj1,adj1_inverse)
        ii = tree.query_ball_point(nodes1.loc[nodes1.index[i]].values, r+1)
        z_list=nodes2.index[ii].values
        dd = norm(nodes1.values[i]-nodes2.loc[z_list].values,axis=1)
        z_i=find_z_heading(dd,x_heading,z_list,nodes2,adj2,adj2_inverse,threshold=threshold)
        if not np.isnan(z_i):
            association_temp.append((nodes1.index[i],z_i,norm(nodes1.values[i]-nodes2.loc[z_i].values)))
        else:
            association_temp.append((nodes1.index[i],z_i,r+1))
    association=pd.DataFrame(association_temp,columns=['ID_query','ID_graph','distance'])
    if anti_loop:
        print("ANTILOOP")
        association=delete_circular_Association(association,nodes2,adj2,nodes1,adj1)
    return association.dropna()


def find_z(x_next_list,z_list,nodes,adj,adj_inverse):
    for x_next in x_next_list:
        i=0
        for each in z_list:
            if (each not in adj.index) or (each not in adj_inverse.index):
                return each, i
            ID_z_next_list=adj.loc[each].values
            ID_z_pre_list=adj_inverse.loc[each].values
            for ID_z_next in ID_z_next_list:
                for ID_z_pre in ID_z_pre_list:
                    z_next=nodes.loc[ID_z_next].values
                    z_pre =nodes.loc[ID_z_pre].values
                    if norm(x_next-z_next)<=norm(x_next-z_pre):
                        return each, i
            i+=1
    return z_list[-1],len(z_list)-1


def revisited(query,adj,depth=10):
    """
    return 2 paths with the same start & end points
    input:
        query: an ID
        adj: adj
        depth:searching depth
    return 2 paths
    """
    que=[]
    visited=[]
    path=[]
    depth_c=0
    que.append((query,None ,0))
    reverse_tree=dict()
    while que:
        
        current,parent,depth_new=que.pop()
        
        if isinstance(current, np.ndarray):
            current=current[0]
        
        
        if depth_new <=depth_c:
            del path[depth_new:]
        depth_c=depth_new
        
        if current in visited:
            backward=current
            bque=[]
            bque.append(current)
            while backward not in path:
                backward=bque.pop()
                bque.extend(reverse_tree[backward])
            path.append(current)
            path_alt=find_path(current,backward,reverse_tree)
            return path[path.index(path_alt[-1]):],path_alt[::-1]
        
        visited.append(current)
        if current in reverse_tree:
            reverse_tree[current].append(parent)
        else:
            reverse_tree[current]=[parent]
        path.append(current)

        if depth_c!=depth:
            if current in adj.index:
                que.extend([(each,current, depth_c + 1) for each in adj.loc[current].values])
    return False

def find_path(start,end,graph):
    """
    return a path from start to end
    """
    path=[]
    que=[]
    que.append((start,0))
    depth_c=0
    while que:
        current, depth_new=que.pop()
        if depth_new <=depth_c:
            del path[depth_new:]
        path.append(current)
        depth_c=depth_new
        if current==end:
            return path
        
        if current in graph:
            que.extend([ (child,depth_new+1) for child in graph[current]])
    return False

##########################################################
def Association_original(query,graph,k=5):
    nodes1,adj1=query.exportWarping()
    nodes2,adj2=graph.exportWarping()
    len_query=nodes1.shape[0]
    
    adj2_inverse=adj2.reset_index().set_index('to_id')
    tree = cKDTree(nodes2)
    dd, ii = tree.query(nodes1, k=k)
    association=pd.DataFrame(columns=['ID_query','ID_graph','distance'])
    association['ID_query']=nodes1.index
    z_list=[]
    for i in range(len_query):
        if nodes1.index[i] in adj1.index:  
            z, i_z=find_z(nodes1.loc[adj1.loc[nodes1.index[i]].values.ravel()].values,nodes2.index[ii[i]].values,nodes2,adj2,adj2_inverse)
            if np.isnan(i_z):
                z_list.append((np.nan,i_z))
            else:
                z_list.append((z,dd[i][i_z]))
        else:
            z_list.append((nodes2.index[ii[len_query-1][0]],dd[len_query-1][0]))
    association[['ID_graph','distance']]=z_list
    return association

def Association(query,graph,k=5,heading=True,r=20,threshold=0.5):
    """
    get association from query to graph
        
    ID_1 is the query ids
    ID_2 is the base graph ids
    """
    if not heading:
        return Association_original(query,graph,k)
    else:
        return Association_heading(query,graph,r,threshold)

    
###############################
## loop detection (only one direction) TODO implemment backwards analogously

def delete_circular_Association(association,nodes_add,adj_add,nodes_base,adj_base):
    uncon_list_forward = get_unconlist(association,nodes_add,adj_add,nodes_base,adj_base)
    adj_base_reverse=adj_base.reset_index().rename(columns={"from_id":"to_id","to_id":"from_id"}).set_index("from_id")
    adj_add_reverse = adj_add.reset_index().rename(columns={"from_id":"to_id","to_id":"from_id"}).set_index("from_id")
    uncon_list_backward = get_unconlist(association,nodes_add,adj_add_reverse,nodes_base,adj_base_reverse)
    uncon_list = pd.concat([uncon_list_forward, uncon_list_backward], axis=0).drop_duplicates()
    ############################################################################################
    delete = True
    while delete:
        half1 = uncon_list.reset_index()[['ID_QUERY']]
        half2 = uncon_list.reset_index()[['ID_QUERY_graph_forw']].rename(columns={"ID_QUERY_graph_forw":"ID_QUERY"})
        counter = pd.concat([half1,half2] ,axis=0)
        cc = counter['ID_QUERY'].value_counts()
        top = cc.head(1).index
        if top.shape[0] > 0:
            malicous = top[0]
            print(malicous)
            association = association[association['ID_query'] != malicous]
            uncon_list = uncon_list[uncon_list['ID_QUERY'] != malicous]
            uncon_list = uncon_list[uncon_list['ID_QUERY_graph_forw'] != malicous]
        else:
            delete = False
        #delete = False
    ############################################################################################
    return association

def get_unconlist(association,nodes_add,adj_add,nodes_base,adj_base):
    ## for all xi, get association zi for xi
    z = nodes_add.join(association.set_index('ID_query'))#[['ID_graph']]
    z = z.reset_index().rename(columns={"id":"ID_QUERY"})[['ID_QUERY','ID_graph']]
    #print(z.head())

    ## for all zi get set of range forward
    graph_range = 5
    # make adjecency matrix reflexive
    from_ids = adj_base.reset_index()[['from_id']]
    reflection = pd.concat([from_ids, from_ids.rename(columns={"from_id":"to_id"})], axis=1)
    refl_adj_base = pd.concat([adj_base.reset_index(), reflection], axis=0)
    #print(refl_adj_base.head())
    #print(refl_adj_base.tail())
    # now get the forward set
    print(z.shape)
    z_forward = z.set_index('ID_graph').join(refl_adj_base.set_index('from_id'))[['ID_QUERY','to_id']]
    print(z_forward.shape)
    #print(z_forward.head())
    for i in range(graph_range):
        #print(i)
        z_forward = z_forward.set_index('to_id').join(refl_adj_base.set_index('from_id'))[['ID_QUERY','to_id']]
        z_forward = z_forward.drop_duplicates()
        print(z_forward.shape)

    print(z_forward.head())
    ############################################################################################
    ## get NN-association in ziforw
    forward = z_forward.set_index('to_id').join(association.set_index('ID_graph')) \
                       .rename(columns={"ID_query":"ID_QUERY_graph_forw"})[['ID_QUERY','ID_QUERY_graph_forw']]
    print(forward.head())
    forward = forward[forward['ID_QUERY'] != forward['ID_QUERY_graph_forw']]
    ############################################################################################
    ### calc ordering (backwards) for 5 steps in sequence
    # make adjecency matrix reflexive
    from_ids = adj_add.reset_index()[['from_id']]
    reflection = pd.concat([from_ids, from_ids.rename(columns={"from_id":"to_id"})], axis=1)
    refl_adj_add = pd.concat([adj_add.reset_index(), reflection], axis=0)
    #print(refl_adj_base.head())
    #print(refl_adj_base.tail())
    #get transpose of adjacency matrix
    refl_adj_add_trans = refl_adj_add.rename(columns={"from_id":"to_id","to_id":"from_id"})
    x_backward = refl_adj_add_trans
    #print(x_backward.head())
    for i in range(graph_range):
    #    #print(i)
        x_backward = x_backward.set_index('to_id').join(refl_adj_add_trans.set_index('from_id')) #[['ID_QUERY','to_id']]
        x_backward = x_backward.drop_duplicates()
        print(x_backward.shape)
    print(x_backward.head())
    ############################################################################################
    ### DETECT if "calc ordering (backwards) for 5 steps in sequence" occurs in "NN-association in ziforw"
    forbidden = x_backward.rename(columns={"from_id":"ID_QUERY","to_id":"ID_QUERY_graph_forw"})
    print(forbidden)
    uncon_list = pd.merge(forward, forbidden, how='inner', on=['ID_QUERY', 'ID_QUERY_graph_forw'])
    uncon_list = uncon_list.astype('int64', copy=False).drop_duplicates()
    return uncon_list