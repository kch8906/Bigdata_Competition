import pandas as pd
import numpy as np
import faiss
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict


def get_pca_data(data,n_components):
    pca = PCA(n_components=n_components)
    pca.fit(data)

    return pca.transform(data), pca
    
def get_pca_df(pca_data,pca):
    cols = ['pca_'+str(1+i) for i in range(pca.components_.shape[0])] 
    
    return pd.DataFrame(pca_data,columns=cols)

def get_vector(idx): 
    return pd.Series(df_x_y.iloc[idx].to_numpy())[0], pd.Series((df_x_y.iloc[idx].to_numpy())[1:]) 


if __name__ == '__main__':

    '''
    데이터 전처리
    '''
    df_01 = pd.read_csv('../3.홍홍홍_데이터 및 모델 세이브 파일/dataset/LPOINT_BIG_COMP/LPOINT_BIG_COMP_01_DEMO.csv')
    df_02 = pd.read_csv('../../3.홍홍홍_데이터 및 모델 세이브 파일/datasetdataset/LPOINT_BIG_COMP/LPOINT_BIG_COMP_02_PDDE.csv')
    df_03 = pd.read_csv('../../3.홍홍홍_데이터 및 모델 세이브 파일/datasetdataset/LPOINT_BIG_COMP/LPOINT_BIG_COMP_03_COP_U.csv')

    df_02_03 = pd.merge(df_02, df_03, on='pd_c')
    df_02_03.drop(['rct_no','cop_c','br_c','pd_c','pd_nm','clac_mcls_nm'], axis=1, inplace=True)
    df_02_03_01 = pd.merge(df_02_03, df_01, on='cust')

    df_x = df_02_03_01.drop('cust', axis=1)
    df_y = df_02_03_01['cust']
    df_one_hot_x = pd.get_dummies(df_x)

    '''
    PCA 차원 축소
    '''
    skca_pca10, pca10 = get_pca_data(df_one_hot_x, 10)
    print(np.sum(pca10.explained_variance_ratio_))


    df_pca_x = pd.DataFrame(skca_pca10)
    df_x_y = pd.concat([df_y, df_pca_x], axis=1)
    df_x_y

    '''
    FAISS 유사도(GPU)
    '''
    vector = []
    for i in tqdm(range(len(df_x_y))):
        _, vec = get_vector(i)
        vector.append(np.array(vec))
        
    vector = np.array(vector)
    vector = vector.astype('float32')

    res = faiss.StandardGpuResources()
    index_flat = faiss.IndexFlatL2(vector.shape[1])
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    gpu_index_flat.add(vector)

    distances, indices = gpu_index_flat.search(vector, 11)

    '''
    
    '''
    result = defaultdict(int)

    for num in tqdm(range(len(df_x_y))):
        pname = df_x_y.iloc[num,:]['cust']
        similar = []
        for idx, loc in enumerate(indices[num][1:]):
            similar.append((df_02_03_01.iloc[loc,:]['clac_hlv_nm'], idx + 1))
        result[pname] = similar

    df_similar = pd.DataFrame(columns=['session', 'similar_item', 'similar_rank'])

    for k, v in tqdm(result.items()):
        for p in v:
            tmp = [k]
            tmp.extend(list(p))
            df_len = len(df_similar)
            df_similar.loc[df_len] = tmp

    df_similar.to_csv('../../3.홍홍홍_데이터 및 모델 세이브 파일/dataset/big_comp/faiss_cold_rank.csv', sep='\t', index=False)
