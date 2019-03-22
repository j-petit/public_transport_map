"""File author: Georg Aures"""
import pymysql
import pdb
import pandas as pd
import numpy as np
import math

MIN_INT = -2000000000
MAX_INT = +2000000000


class DBInterface(object):
    @staticmethod
    def _column_names():
        """ provides and defines the names of the dataframe columns, renaming possible here.
        """
        return ['ABS_VEH_ODOMETRY','VEH_TIME','TIME_DURATION', 'VEH_Y', 'VEH_X', 'GPS_QUALITY'\
                , 'DOOR', 'VEH_GPS_ID', 'D_VEH_ODOMETRY', 'D_TIME','GPS_CONSISTENCY']
    
    @staticmethod
    def __SELECT_FROM_statement():
        """Defines the FROM part for the SQL query.
        """
        inner_query = """SELECT VEH_GPS_ID
                               ,VEH_ADRESS
                               ,VEH_TIME
                               ,VEH_ODOMETRY
                               ,VEH_LAT
                               ,VEH_LON
                               ,GPS_QUALITY
                               ,case when EXISTS (SELECT 1 
                                                    FROM tbl_VEH_DOOR_new as VDOOR
                                                   WHERE VDOOR.VEH_ADDRESS = tbl_VEH_GPS.VEH_ADRESS
                                                     AND VDOOR.VEH_GPS_ID = tbl_VEH_GPS.VEH_GPS_ID)
                                     then 1 else 0 end 
                                as DOOR
                          FROM qry_VEH_TRIP_full
                          JOIN tbl_VEH_GPS
                            ON qry_VEH_TRIP_full.VEH_ADDRESS = tbl_VEH_GPS.VEH_ADRESS
                           AND tbl_VEH_GPS.VEH_TIME BETWEEN qry_VEH_TRIP_full.ACT_DEP_TIMESTAMP 
                                                        and qry_VEH_TRIP_full.ACT_END_TIMESTAMP
                    """
        return inner_query
    
    @staticmethod
    def _fetchTRIP(VEH_TRIP_ID, debug=False):
        """Defines the WHERE clause of the qry. holds for the passed VEH_TRIP_ID.

        Parameters:
            VEH_TRIP_ID : Number in SQL database specifying the trip
            debug       : boolean if true print messages are enabled
        """
        
        qry_where = DBInterface.__SELECT_FROM_statement() + """ WHERE VEH_TRIP_ID = """ + str(VEH_TRIP_ID)

        return DBInterface._fetchGeneric(qry_where,debug)

    @staticmethod
    def _fetchBLOCK(VEH_BLOCK_ID, debug=False):
        """Defines the WHERE clause of the qry. holds for the passed VEH_BLOCK_ID.
        A block (with unique VEH_BLOCK_ID) is a set of trips (with unique VEH_TRIP_ID)

        Parameters:
            VEH_BLOCK_ID : Number in SQL database specifying the block
            debug        : boolean if true print messages are enabled
        """

        qry_where = DBInterface.__SELECT_FROM_statement() + """ WHERE VEH_BLOCK_ID = """ + str(VEH_BLOCK_ID)

        return DBInterface._fetchGeneric(qry_where,debug)

    ###TODO write bounded qry
    ###WHERE VEH_LAT between 48.086049 and 48.091302
    #### and VEH_LON between 11.476640 and 11.484440
    
    @staticmethod
    def _fetchGeneric(inner_query, debug=False):
        """Defines the SELECT part for the SQL query.
        matches DOOR data to the GPS data and groups by odometry.
        returns the Data in a pandas Dataframe if minimum data integrity is given

        Parameters:
            qry_where : the inner query containing the where clause
        """

        if debug:
            print(inner_query)

        try:
            # Connect to database,
            db_mvg_data = pymysql.connect(host='localhost',
                                          user='root',
                                          passwd='mllab2018',
                                          db='db_mvg_data')

            # check integrity
            if DBInterface.__integritity_fulfilled(inner_query, db_mvg_data):
                if debug:
                    print('minimum integrity fulfilled')
            else:
                return pd.DataFrame(columns=DBInterface._column_names()) # pd.DataFrame.empty
                if debug:
                    print('CORRUPT DATA')

            # fetch DATA,
            query_GPS_DATA = """SELECT VEH_ODOMETRY
                                   ,MIN(UNIX_TIMESTAMP(VEH_TIME)) AS VEH_TIME
    ,MAX(UNIX_TIMESTAMP(VEH_TIME))-MIN(UNIX_TIMESTAMP(VEH_TIME)) AS TIME_DURATION
    ,SUM(IFNULL(VEH_LAT,0)*POWER(IFNULL(GPS_QUALITY,0),2))*111192.981461485/NULLIF(SUM(POWER(IFNULL(GPS_QUALITY,0),2)),0) as VEH_Y
    ,SUM(IFNULL(VEH_LON,0)*POWER(IFNULL(GPS_QUALITY,0),2))*74427.2442617538/NULLIF(SUM(POWER(IFNULL(GPS_QUALITY,0),2)),0) as VEH_X
                                  ,MAX(GPS_QUALITY) as GPS_QUALITY
                                  ,MAX(DOOR) as DOOR
                                  ,MAX(VEH_GPS_ID) as VEH_GPS_ID
                              FROM (""" + inner_query + """) as innerQuery
                            GROUP BY VEH_ODOMETRY
                            """

            GPS_DATA = pd.read_sql(query_GPS_DATA, db_mvg_data)

            # disconnect from database
            db_mvg_data.close()

            return GPS_DATA
        except:
            db_mvg_data.close()
        return pd.DataFrame(columns=DBInterface._column_names()) # pd.DataFrame.empty
    
    @staticmethod
    def __integritity_fulfilled(inner_query, db_mvg_data):
        """returns true if the odometry is consistent and there exists GPS data for the trip

        Parameters:
            inner_query : the inner query containing the where clause
            db_mvg_data : the database conection
        """
                # fetch integrity,
        query_integrity = "SELECT VEH_ODOMETRY FROM (" + inner_query + \
        ") as innerQuery ORDER BY VEH_TIME, VEH_ODOMETRY "
                
        ODO_INTEGRITY = pd.read_sql(query_integrity, db_mvg_data)
            # check integrity
        integrity = (pd.Index(ODO_INTEGRITY['VEH_ODOMETRY']).is_monotonic and ODO_INTEGRITY.size > 9)

        return integrity


    @staticmethod
    def _interpolateData(GPS_DATA, raw=False, outlier_param=0.8, debug=False, min_sparse=None):
        """interpolates the GPS_DATA at missing GPS positions
          ,detects outliers and interpolates them to if raw is true

        Parameters:
            GPS_DATA      : the GPS_DATA fetched beforhand
            raw           : boolean if true outliers are interpolated too
            outlier_param : $$\in \(0,1\]$$ defining consistency_values smaller
                            than outlier_param as outliers
            debug         : if true print debug messages
        """
        lat_divide = 111192.981461485
        lon_divide = 74427.2442617538
        ## remove data outside munich
        GPS_DATA.loc[GPS_DATA['VEH_Y'] < 47.5*lat_divide, ['GPS_QUALITY','VEH_Y','VEH_X']] = [0,np.nan,np.nan]
        GPS_DATA.loc[GPS_DATA['VEH_X'] < 11.0*lon_divide, ['GPS_QUALITY','VEH_Y','VEH_X']] = [0,np.nan,np.nan]
        GPS_DATA.loc[GPS_DATA['VEH_Y'] > 48.5*lat_divide, ['GPS_QUALITY','VEH_Y','VEH_X']] = [0,np.nan,np.nan]
        GPS_DATA.loc[GPS_DATA['VEH_X'] > 12.0*lon_divide, ['GPS_QUALITY','VEH_Y','VEH_X']] = [0,np.nan,np.nan]
        
        if GPS_DATA['VEH_Y'].isnull().all() == False:
            ## interpolate DATA
            GPS_DATA[['VEH_Y','VEH_X']] = GPS_DATA[['VEH_Y','VEH_X']].interpolate(method='linear',limit_direction='both')
            ## calc diff in odometry
            shift_forw = GPS_DATA[['VEH_ODOMETRY','VEH_TIME']].shift(-1)
            GPS_DATA[['D_VEH_ODOMETRY']] = shift_forw[['VEH_ODOMETRY']] - GPS_DATA[['VEH_ODOMETRY']]
            GPS_DATA[['D_TIME']]=shift_forw[['VEH_TIME']]-GPS_DATA[['VEH_TIME']]
            ## mark ouliers
            error_fact1 = (DBInterface.__error_fact(GPS_DATA, step = 1))
            error_fact = ( DBInterface.__error_fact(GPS_DATA, step = 2).fillna(error_fact1) \
                          +DBInterface.__error_fact(GPS_DATA, step = 3).fillna(error_fact1) \
                          +DBInterface.__error_fact(GPS_DATA, step = 5).fillna(error_fact1) \
                          +DBInterface.__error_fact(GPS_DATA, step = 7).fillna(error_fact1) \
                         )/4
            GPS_DATA['GPS_CONSISTENCY'] = (1 / (1 + error_fact)).fillna(0)

            if (not(raw)):
                # remove outliers larger than outlier_param
                GPS_DATA.loc[GPS_DATA['GPS_CONSISTENCY'] < outlier_param, ['GPS_QUALITY','VEH_Y','VEH_X']] = [0,np.nan,np.nan]
                ## interpolate DATA
                GPS_DATA[['VEH_Y','VEH_X']] = GPS_DATA[['VEH_Y','VEH_X']].interpolate(method='linear',limit_direction='both')

            if min_sparse is not None:
                GPS_DATA = DBInterface.brute_sparse(GPS_DATA, min_sparse, direction='both', debug=debug)
                GPS_DATA = DBInterface.brute_sparse(GPS_DATA, min_sparse, direction='one', debug=debug)
                # update D_VEH_ODOMETRY and D_TIME
                shift_forw = GPS_DATA[['VEH_ODOMETRY', 'VEH_TIME']].shift(-1)
                GPS_DATA[['D_VEH_ODOMETRY']] = shift_forw[['VEH_ODOMETRY']] - GPS_DATA[['VEH_ODOMETRY']]
                GPS_DATA[['D_TIME']] = shift_forw[['VEH_TIME']] - GPS_DATA[['VEH_TIME']]

            return GPS_DATA
        else:
            return None

    @staticmethod
    def brute_sparse(GPS_DATA, min_sparse, direction='both', debug=False):
        """iteratively deletes every second point in GPS_DATA
            that is closer than min_sparse to 'both'/'one' of
            its neighbours.
            returns GPS_DATA where D_VEH_ODOMETRY >= min_sparse

            Parameters:
            GPS_DATA      : the GPS_DATA fetched beforhand
            min_sparse    : the minimum D_VEH_ODOMETRY -- if violated values get dropped
            direction     : checks violation of min_sparse in 'both' or 'one' direction
            debug         : if True print debug messages
        """
        found_close_points = True
        while found_close_points:
            shift_forw_upd = GPS_DATA[['VEH_ODOMETRY']].shift(-1).fillna(MAX_INT)
            shift_backw_upd = GPS_DATA[['VEH_ODOMETRY']].shift(+1).fillna(MIN_INT)
            sparse1 = shift_forw_upd - GPS_DATA[['VEH_ODOMETRY']]
            sparse2 = GPS_DATA[['VEH_ODOMETRY']] - shift_backw_upd

            if direction == 'both':
                list = sparse1[(sparse1['VEH_ODOMETRY'] < min_sparse)
                               & (sparse2['VEH_ODOMETRY'] < min_sparse)
                               & (GPS_DATA['DOOR'] == 0)].index.values
            else:
                list = sparse1[((sparse1['VEH_ODOMETRY'] < min_sparse)
                               | (sparse2['VEH_ODOMETRY'] < min_sparse))
                               & (GPS_DATA['DOOR'] == 0)].index.values

            if debug:
                print(list)

            list = list[0::2]

            if debug:
                print(list)
                print('#########')

            if len(list) > 0:
                GPS_DATA = GPS_DATA.drop(list)
            else:
                found_close_points = False

        return GPS_DATA

    @staticmethod
    def __error_fact(GPS_DATA, step = 1, min_error_param = 3.0):
        """calculating the weighted difference between gps-distance and odometry minus min_error_param
           then dividing by odometry to get result as factor
        """
        shift_forw = GPS_DATA[['VEH_Y','VEH_X','VEH_ODOMETRY','GPS_QUALITY']].shift(-step)
        shift_back = GPS_DATA[['VEH_Y','VEH_X','VEH_ODOMETRY','GPS_QUALITY']].shift(+step)
        # calc
        weight_sb = shift_back.GPS_QUALITY.fillna(0)
        weight_sf = shift_forw.GPS_QUALITY.fillna(0)
        error_sb1 = DBInterface.__dist_error(GPS_DATA, shift_back) * weight_sb
        error_sf1 = DBInterface.__dist_error(shift_forw, GPS_DATA) * weight_sf
        sum_weights = (weight_sb + weight_sf) \
                    * ( shift_forw.VEH_ODOMETRY.fillna(GPS_DATA.VEH_ODOMETRY) \
                       -shift_back.VEH_ODOMETRY.fillna(GPS_DATA.VEH_ODOMETRY) \
                      ).fillna(1)
        error_fact = (error_sb1 + error_sf1).fillna(0) / sum_weights
        return error_fact
    
    @staticmethod
    def __dist_error(GPS_DATA, shift_back, min_error_param = 3.0):
        """calculating the difference between gps-distance and odometry minus min_error_param
           only report positive errors, that means gps-distance is larger than odometry
        """
        # if forward please call __dist_error(shift_forw, GPS_DATA)
        euclid = ((shift_back.VEH_Y - GPS_DATA.VEH_Y) ** 2 + (shift_back.VEH_X - GPS_DATA.VEH_X) ** 2 ) ** .5
        odometry = GPS_DATA.VEH_ODOMETRY - shift_back.VEH_ODOMETRY
        return (euclid - odometry - min_error_param).clip(0, None).fillna(0) 

    @staticmethod
    def getTripsForRouteAndDay(line, op_date, route='%',debug=False):
        """get the TRIP_IDs for a given route and day
           see :func:`getTripsForDay <db_interface.DBInterface.getTripsForDay>`
            
            Parameters:
            line          : String defining the line eg. '166'
            op_date       : operational day eg '2018-04-14'
            route         : String defining the route eg. '1'
            debug         : if true print debug messages    
        """
        try:
            qry_for_code = """select VEH_TRIP_ID, PATTERN_CODE from qry_VEH_TRIP_full_plus_pattern 
                               where PATTERN_CODE like '_:{0}:{1}' and OP_DATE = '{2}';"""
            qry_for_code = qry_for_code.format(line, route, op_date)
            if debug:
                print(qry_for_code)

            ## Connect to database,
            db_mvg_data = pymysql.connect(host='localhost',
                                          user='root',
                                          passwd='mllab2018',
                                          db='db_mvg_data')

            ## fetch corresponding trip IDs
            VEH_TRIP_IDs = pd.read_sql(qry_for_code, db_mvg_data)

            ##  disconnect from database
            db_mvg_data.close()

            return VEH_TRIP_IDs
        except:
            db_mvg_data.close()
        return pd.DataFrame.empty

    @staticmethod
    def getTripsForDay(op_date, debug=False, add_info=False):
        """get the TRIP_IDs for a given day
           see :func:`getTripsForRouteAndDay <db_interface.DBInterface.getTripsForRouteAndDay>`
            
            Parameters:
            op_date       : operational day eg '2018-02-14'
            debug         : if true print debug messages   
            add_info      : if true return not only DataFrame(VEH_TRIP_ID) 
                            but DataFrame(VEH_TRIP_ID, PATTERN_ID, PATTERN_QUALITY, TRIP_TYPE, TRIP_CODE)
        """
        try:
            if add_info:
                qry_for_code = """select VEH_TRIP_ID, PATTERN_ID, PATTERN_QUALITY, TRIP_TYPE, TRIP_CODE 
                                    from qry_VEH_TRIP_full where OP_DATE = '{0}';"""
            else:
                qry_for_code = """select VEH_TRIP_ID from qry_VEH_TRIP_full where OP_DATE = '{0}';"""
            qry_for_code = qry_for_code.format(op_date)

            ## Connect to database,
            db_mvg_data = pymysql.connect(host='localhost',
                                          user='root',
                                          passwd='mllab2018',
                                          db='db_mvg_data')

            ## fetch corresponding trip IDs
            VEH_TRIP_IDs = pd.read_sql(qry_for_code, db_mvg_data)

            ##  disconnect from database
            db_mvg_data.close()

            return VEH_TRIP_IDs
        except:
            db_mvg_data.close()
        return pd.DataFrame.empty
    
    @staticmethod
    def loadTrip(VEH_TRIP_ID, debug=False, raw=False, outlier_param=0.8, min_sparse=None):
        """load GPS_DATA of a given VEH_TRIP_ID as pandas dataframe
          ,with these columns: 'ABS_VEH_ODOMETRY', 
                              'VEH_Y', 'VEH_X', 
                              'GPS_QUALITY'
                              ,'DOOR', 
                              'VEH_GPS_ID', 
                              'D_VEH_ODOMETRY', 
                              'GPS_CONSISTENCY'

        Parameters:
            VEH_TRIP_ID     : VEH_TRIP_ID see :func:`my text <db_interface.DBInterface.getTripsForRouteAndDay>`
            [debug]         : if true print debug messages
                              default False
            [raw]           : boolean if True outliers are interpolated too
                              default False
            [outlier_param] : $$\in \(0,1\]$$ defining consistency_values smaller
                              than outlier_param as outliers
                              default 0.8
            [min_sparse]    : minimum Odometry between two consecutive measurements,
                              if data contains higher resolution measurements those will be dropped
                              door_events will never bee dropped
                              default None
        """
        Data = DBInterface._fetchTRIP(VEH_TRIP_ID,debug)
        if Data.empty:
            return Data
        interpolatedData = DBInterface._interpolateData(Data
                                                        ,debug=debug
                                                        ,outlier_param=outlier_param
                                                        ,raw=raw
                                                        ,min_sparse=min_sparse)

        # apply renaming here:
        if interpolatedData is not None:
            interpolatedData.columns = DBInterface._column_names()
        return interpolatedData

    @staticmethod
    def loadTripSplit(VEH_TRIP_ID, debug=False, raw=False, outlier_param=0.8, min_sparse=None, split_ratio=0.7,
                      split_on='ABS_VEH_ODOMETRY'):
        """load GPS_DATA of a given VEH_TRIP_ID as two pandas dataframes where
           the first contains the observation_data and the second one the prediction_data
          ,with these columns: 'ABS_VEH_ODOMETRY',
                              'VEH_Y', 'VEH_X',
                              'GPS_QUALITY'
                              ,'DOOR',
                              'VEH_GPS_ID',
                              'D_VEH_ODOMETRY',
                              'GPS_CONSISTENCY'

        Parameters:
            VEH_TRIP_ID     : VEH_TRIP_ID see :func:`my text <db_interface.DBInterface.getTripsForRouteAndDay>`
            [debug]         : if true print debug messages
                              default False
            [raw]           : boolean if True outliers are interpolated too
                              default False
            [outlier_param] : $$\in \(0,1\]$$ defining consistency_values smaller
                              than outlier_param as outliers
                              default 0.8
            [min_sparse]    : minimum Odometry between two consecutive measurements,
                              if data contains higher resolution measurements those will be dropped
                              door_events will never bee dropped
                              default None
            [split_ratio]   : the ratio at with the data is split
            [split_on]      : split on 'VEH_TIME' or  'ABS_VEH_ODOMETRY'
        """
        if split_ratio > 1.0:
            split_ratio = 1.0
        elif split_ratio < 0.0:
            split_ratio = 0.0
        elif not ( 0.0 <= split_ratio and split_ratio <= 1.0 ):
            split_ratio = 1.0

        if not (split_on == 'ABS_VEH_ODOMETRY' or split_on == 'VEH_TIME'):
            split_on = 'ABS_VEH_ODOMETRY'

        wholeData = DBInterface.loadTrip(VEH_TRIP_ID,
                                              debug=debug,
                                              raw=raw,
                                              outlier_param=outlier_param,
                                              min_sparse=min_sparse)

        min_time = wholeData[split_on].min()
        print(min_time)
        max_time = wholeData[split_on].max()
        split_time = math.ceil(min_time + (max_time - min_time) * split_ratio)
        observation_data = wholeData[wholeData[split_on] <= split_time]
        prediction_data = wholeData[wholeData[split_on] > split_time]
        return observation_data, prediction_data