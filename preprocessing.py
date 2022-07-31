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
    global num_cores
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
            frequency[i] = random.uniform(frequency[i] - 1, frequency[i] + 0.1)
        frequency = np.round(frequency, 1)
        
        for i, j in zip(frequency.index, frequency.values):
            if cop_c == i:
                return j
                
    except: ValueError

def copMakeRatings(data):
    data['ratings'] = data.progress_apply(lambda x : convertRatingCop(x['cust'], x['cop_c'], data), axis=1)
    return data