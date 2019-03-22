""" Author: Chengyu Sheu"""
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
import pymysql
import pandas as pd
import collections
import numpy as np
import networkx as nx

#posterior prob of nodes (EKF)
def pos_node(mean,variance,pose_graph,k=6,r=50):
    p_list=list()
    nodes,adj=pose_graph.exportWarping()
    tree=cKDTree(nodes)
    index=tree.query_ball_point((mean), r)
    if len(index)<k:
        dd,index = tree.query(mean,k)
    neibor=nodes.iloc[index].values-mean
    
    for i in range(len(neibor)):
        p=np.exp(neibor[i]@np.linalg.inv(variance)@neibor[i].T/-2)
        p_list.append(p)
    p=np.array(p_list)
    p/=p.sum()
    return nodes.iloc[index].index,p

# BAY DISTRIBUTION
def same_dirc(a,b,threshold=np.pi/2):
    maxi=max(a,b)
    mini=min(a,b)
    diff=min(maxi-mini,mini-maxi+2*np.pi)
    return diff <= threshold
def mydistance(x,y):
    if not same_dirc(x[-1],y[-1]):
        return np.inf
    else:
        return np.sqrt(((x[:-1]-y[:-1])**2).sum())

def get_attr(pose_graph,query='door_counter'):
    attr=nx.get_node_attributes(pose_graph.nx_graph,query)
    return pd.DataFrame.from_dict(attr,orient='index').rename(columns={0:query})

def bay_distribution(pose_graph,eps=3,minpts=1):
    DOOR=get_attr(pose_graph)
    DOOR=DOOR[DOOR['door_counter']!=0].index
    nodes,adj=pose_graph.exportWarping()
    heading=np.array(pose_graph.getNodeHeading([*DOOR]))
    heading=heading.reshape((heading.shape[0],1))
    DOOR=nodes.loc[DOOR]
    DOOR_heading=np.hstack((DOOR.values,heading))
    labels = DBSCAN(eps=eps, min_samples=minpts, metric=mydistance).fit_predict(DOOR_heading)
    means=list()
    variances=list()
    bays=list()
    for label in np.unique(labels):
        if label == -1:
            for each in DOOR.iloc[labels==-1].values:
                means.append(each)
                variances.append(np.diag([np.inf,np.inf]))
            bays.append(DOOR.iloc[labels==-1].index)
        else:
            
            mean = np.mean(DOOR.iloc[labels==label].values, axis=0)
            cov = np.cov(DOOR.iloc[labels==label].values, rowvar=0)
            
            means.append(mean)
            variances.append(cov)
            bays.append(DOOR.iloc[labels==label].index)
    return bays,means,variances,DOOR

def find_label(D,b,graph_merged_final,query_position):
    label=list()
    query=graph_merged_final._to_x_y(query_position)
    tree=cKDTree(D)
    i=tree.query_ball_point(query,50)
    N_index=D.index[i]
    for index in N_index:
        for each in range(len(b)):
            if index in b[each]:
                label.append(each)
    return label

# PATTERN estimation

def get_pattern_db(line):

    db_mvg_data = pymysql.connect(host='localhost',user='root',passwd='mllab2018',db='db_mvg_data')
    qry_random_trip = """select PATTERN_ID,PATTERN_CODE, VALID_FROM 
                        FROM tbl_NOM_PATTERN as original
                        WHERE  original.DIRECTION IS NOT NULL AND original.PATTERN_CODE LIKE 'B:"""\
                        +str(line)+""":%'
                        AND NOT EXISTS(
                            SELECT 1
                            FROM tbl_NOM_PATTERN as newer
                            WHERE newer.VALID_FROM>original.VALID_FROM
                            and newer.PATTERN_NO=original.PATTERN_NO)
                        GROUP BY PATTERN_NO,DIRECTION,PATTERN_CODE,PATTERN_ID;
                        """
    pattern=pd.read_sql(qry_random_trip,db_mvg_data)

    pattern_BAY_all=pd.DataFrame(columns=['BAY_DEF_ID','DIST_TO_START','DIST_TO_NEXT'])
    for each in pattern.index:
        qry_bay="""select BAY_DEF_ID, DIST_TO_START, DIST_TO_NEXT
                    FROM tbl_NOM_PATTERN_BAY
                    WHERE PATTERN_ID="""+str(pattern['PATTERN_ID'][each])
        pattern_BAY=pd.read_sql(qry_bay,db_mvg_data)
        pattern_BAY['PATTERN_ID']=pattern['PATTERN_CODE'][each]
        pattern_BAY_all=pattern_BAY_all.append(pattern_BAY)

    pattern_BAY_all=pattern_BAY_all.reset_index()

    BAY_all=pd.DataFrame(columns=['BAY_DEF_ID', 'x', 'y'])
    for each in pattern_BAY_all['BAY_DEF_ID']:
        qry_position="""select BAY_DEF_ID, BAY_LONGITUDE*74427.2442617538 as x, BAY_LATITUDE*111192.981461485 as y
                    FROM tbl_NOM_BAY as original
                    WHERE NOT EXISTS(
                            SELECT 1
                            FROM tbl_NOM_BAY as newer
                            WHERE newer.VALID_FROM>original.VALID_FROM
                            and newer.BAY_DEF_ID=original.BAY_DEF_ID)
                    AND BAY_DEF_ID="""+str(each)
        pattern_position=pd.read_sql(qry_position,db_mvg_data)
        BAY_all=BAY_all.append(pattern_position)
    BAY_all=BAY_all.reset_index()
    result=pd.DataFrame(columns=['PATTERN_ID','BAY_ID','x','y'])
    result['x']=BAY_all['x']
    result['y']=BAY_all['y']
    result['PATTERN_ID']=pattern_BAY_all['PATTERN_ID']
    result['BAY_ID']=pattern_BAY_all['BAY_DEF_ID']
    return result  

def Bay_position(A):
    A=A.set_index('BAY_ID')
    return A.loc[A.index.unique()].max(level=0)[['x','y']]

def P2B(pattern_ID,pattern_bay_all):
    """
    P2B('B:52:1',pd.DataFrame(columns=['PATTERN_ID','BAY_ID']))
    """
    return pattern_bay_all.set_index('PATTERN_ID').loc[pattern_ID]

def B2P(bays,pattern_bay_all,r=10):
    """
    B2P([2787063,2786959],pd.DataFrame(columns=['PATTERN_ID','BAY_ID']))
    """
    bay_all=Bay_position(pattern_bay_all)
    tree=cKDTree(bay_all)
    index=tree.query_ball_point(bays,r=r)
    flat=[]
    for each in index:
        flat+=each
    flat=np.array(bay_all.index[flat])
    Bay_Count=collections.Counter(flat)
    Pattern_Count=collections.Counter()
    for each in Bay_Count:
        Pattern=collections.Counter(pattern_bay_all.set_index('BAY_ID').loc[each]['PATTERN_ID'])
        for k in Pattern:
            Pattern[k]*=Bay_Count[each]
        Pattern_Count+=Pattern
    
    Pattern_pd=pd.DataFrame.from_dict(Pattern_Count,orient='index').rename(columns={0:'probability'})
    Pattern_likly=Pattern_pd/Pattern_pd.sum()
    
    return Pattern_likly