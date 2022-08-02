import pandas as pd
import numpy as np
import multiprocessing as mp
import random
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
tqdm.pandas()


'''
멀티프로세싱
판다스 데이터 병렬 처리(CPU)
'''
def parallel_dataframe(df, func, cores):
    df_split = np.array_split(df, cores)
    pool = mp.Pool(cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    
    return df


'''
제휴사 데이터
파생변수(ratings) 생성
'''

def convertRatingCop(cust, cop_c, df):
    try:
        frequency = df[df['cust'] == cust]['cop_c']
        frequency = frequency.value_counts(normalize=True) * 10
        for i in range(len(frequency)):
            frequency[i] = random.uniform(frequency[i] - 1, frequency[i])
        frequency = np.round(frequency, 2)
        
        for i, j in zip(frequency.index, frequency.values):
            if cop_c == i:
                res = np.abs(j)
                return res
                
    except: ValueError


def copMakeRatings(data):
    data['ratings'] = data.progress_apply(lambda x : convertRatingCop(x['cust'], x['cop_c'], data), axis=1)
    return data


'''
상품 데이터
파생변수(ratings) 생성
'''

def convert_rating(cust, clac_hlv_nm, df):  
    try:
        pd_c = df[df['cust'] == cust][['clac_hlv_nm', 'buy_ct']]
        pd_c = pd_c.groupby('clac_hlv_nm')['buy_ct'].sum().reset_index()
        pd_c = pd_c.sort_values(by='buy_ct', ascending=False, ignore_index=True)

        pd_c['buy_ct'] = (pd_c['buy_ct'] - pd_c['buy_ct'].min()) / (pd_c['buy_ct'].max() - pd_c['buy_ct'].min())
        pd_c['buy_ct'][0] = random.uniform(0.9, 1.01) 
        pd_c['buy_ct'] = np.round(pd_c['buy_ct'], 3) * 10

        res = float(pd_c[pd_c['clac_hlv_nm'] == clac_hlv_nm]['buy_ct'])
   
    except: ValueError
        
    return res

def make_ratings(data):
    data['ratings'] = data.progress_apply(lambda x : convert_rating(x['cust'], x['clac_hlv_nm'], data), axis=1)
    return data