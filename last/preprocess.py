import pandas as pd
import warnings
import numpy as np
import random
import time
import multiprocessing as mp
import datetime as dt
from datetime import datetime
from tqdm import tqdm
warnings.filterwarnings('ignore')
tqdm.pandas()

def parallel_dataframe(df, func):  
        global num_cores
        df_split = np.array_split(df, num_cores)
        pool = mp.Pool(num_cores)
        df = pd.concat(pool.map(func, df_split))
        pool.close()
        pool.join()        
        return df

def convert_timestamp(de_dt, de_hr, df):    
    hourstamp = 60 * 60 * de_hr  
    res = time.mktime(dt.datetime.strptime(de_dt, "%Y-%m-%d").timetuple()) + hourstamp            
    return res

def make_timestamp(df):
    df['timestamp'] = df.progress_apply(lambda x : convert_timestamp(x['de_dt'], x['de_hr'], df), axis=1)
    return df

def preprocess():    
    df_02 = pd.read_csv('../../dataset/LPOINT_BIG_COMP/LPOINT_BIG_COMP_02_PDDE.csv')    
    df_04 = pd.read_csv('../../dataset/LPOINT_BIG_COMP/LPOINT_BIG_COMP_04_PD_CLAC.csv')    

    merge_02_04 = pd.merge(df_02, df_04, on='pd_c')
    convert_time_df = merge_02_04[['cust', 'clac_hlv_nm', 'de_dt', 'de_hr']]

    convert_time_df['de_dt'] = convert_time_df['de_dt'].astype('str')
    convert_time_df['de_dt'] = pd.to_datetime(convert_time_df['de_dt'])
    convert_time_df['de_dt'] = convert_time_df['de_dt'].astype('str')

    data = parallel_dataframe(convert_time_df, make_timestamp)
    data.drop(['de_dt', 'de_hr'], axis=1, inplace=True)
    data = data[['cust', 'timestamp', 'clac_hlv_nm']]

    data.columns = ['session', 'timestamp', 'item']

    data.to_csv('../../dataset/big_comp/convert_time_df.csv', sep='\t', index=False)

if __name__ == '__main__':
    num_cores = mp.cpu_count()
    preprocess()