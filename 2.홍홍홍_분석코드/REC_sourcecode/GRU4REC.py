import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from tqdm import tqdm


class SessionDataset:

    def __init__(self, df):

        self.df = df.sort_values(by = ['session', 'timestamp']).reset_index(drop = True) # session (int) | timestamp (int) | item (string)
        self.offsets = np.concatenate((np.zeros(1, dtype = np.int32), self.df.groupby('session').size().cumsum().values)) 
        self.n_sessions = len(self.offsets) - 1

        self.item_to_id = {item : i for i, item in enumerate(self.df.item.unique())}

        self.n_items = len(self.item_to_id)

    def item_to_one_hot(self, item):
        
        return tf.one_hot(self.item_to_id[item], depth = self.n_items)

    def extract_session(self, i, one_hot_encoded = True):
        session = self.df[self.offsets[i]:self.offsets[i+1]].copy()
        
        if one_hot_encoded:
            session.loc[:, 'item'] = session.item.apply(lambda x : self.item_to_one_hot(x))
        return session.item.values.tolist()


class Gru4Rec:

    def __init__(self, n_classes, n_layers = 1, n_hidden = 64, loss = TOP1, batch_size = 8):

        self.n_classes  = n_classes   

        self.n_layers = n_layers  
        self.n_hidden = n_hidden  
        self.loss     = loss
        self.batch_size = batch_size

        self.model = self.build_model()

    def build_model(self):

        model = tf.keras.models.Sequential()
        for i in range(self.n_layers):
            model.add(tf.keras.layers.GRU(name = 'GRU_{}'.format(i+1),
                                          units      = self.n_hidden, 
                                          activation = 'relu', 
                                          stateful   = True,
                                          return_sequences = (i < self.n_layers - 1)))
        model.add(tf.keras.layers.Dense(units = self.n_classes, activation = 'linear'))  

        top10accuracy = lambda y_true, y_pred: tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k = 10)
        top10accuracy.__name__ = 'top10_accuracy'
        model.compile(loss = self.loss, optimizer = 'adam', metrics = ['accuracy', top10accuracy])

        model.build(input_shape = (self.batch_size, 1, self.n_classes))
        print(model.summary())

        return model

    def _reset_hidden(self, i):

        for nl, layer in enumerate(self.model.layers):   
            if self._is_GRU_layer(layer) and layer.states[0] is not None:
                hidden_updated = layer.states[0].numpy()
                hidden_updated[i, :] = 0.
                self.model.layers[nl].reset_states(hidden_updated)

    def _is_GRU_layer(self, layer):

        return layer.name.startswith('GRU_')

    def train_batch_generator(self, dataset):  # session | item | timestamp

        assert dataset.n_sessions > self.batch_size, 
        ixs = np.arange(dataset.n_sessions)

        stacks = [[]] * self.batch_size
        next_session_id = 0

        X, y = np.empty(shape = (self.batch_size, 1, self.n_classes)), np.empty(shape = (self.batch_size, self.n_classes))    
        while True:
            X[:], y[:] = None, None
            for i in range(self.batch_size): 
                
                if len(stacks[i]) <= 1:
                    if next_session_id >= dataset.n_sessions: 
                        np.random.shuffle(ixs)
                        next_session_id = 0
                    while not len(stacks[i]) >= 2:   
                        stacks[i] = dataset.extract_session(ixs[next_session_id])[::-1] 
                        
                        next_session_id += 1
                    self._reset_hidden(i)   
                
                X[i, 0] = stacks[i].pop()
                y[i]    = stacks[i][-1]

            yield tf.constant(X, dtype = tf.float32), tf.constant(y, dtype = tf.float32)

    def fit(self, dataset, steps_per_epoch = 10000, epochs = 5):

        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath = "gru-chkpt-{epoch:02d}.h5")
        self.model.fit_generator(generator       = self.train_batch_generator(dataset), 
                                 steps_per_epoch = steps_per_epoch, 
                                 epochs          = epochs,
                                 callbacks       = [checkpoint], 
                                 shuffle         = False)


if __name__ == '__main__':
    sampling = False

    if sampling: 
        
        def BPR(y_true, y_pred):
            to_lookup = tf.argmax(y_true, axis = 1) 
            scores = tf.nn.embedding_lookup(tf.transpose(y_pred), to_lookup) 
            return tf.reduce_mean(-tf.math.log(tf.nn.sigmoid(tf.linalg.diag_part(scores) - scores)))

        def TOP1(y_true, y_pred):
            to_lookup = tf.argmax(y_true, axis = 1)
            scores = tf.nn.embedding_lookup(tf.transpose(y_pred), to_lookup)
            diag_scores = tf.linalg.diag_part(scores)
            loss_by_sample  = tf.reduce_mean(tf.nn.sigmoid(scores - diag_scores) + tf.nn.sigmoid(tf.square(scores)), axis = 0)
            loss_by_sample -= tf.nn.sigmoid(tf.square(diag_scores)) / tf.reduce_sum(tf.ones_like(diag_scores)) 
            return tf.reduce_mean(loss_by_sample)

    else: 

        def BPR(y_true, y_pred):  
            _y_pred = tf.expand_dims(y_pred, axis = -1)  
            mat = tf.matmul(tf.expand_dims(tf.ones_like(y_true), -1), tf.expand_dims(y_true, axis = 1)) 
            score_diffs = tf.matmul(mat, _y_pred) 
            score_diffs = tf.squeeze(score_diffs - _y_pred, -1) 
            return -tf.reduce_sum(tf.math.log(tf.nn.sigmoid(score_diffs)))

        def TOP1(y_true, y_pred):
            _y_pred = tf.expand_dims(y_pred, axis = -1)  
            mat = tf.matmul(tf.expand_dims(tf.ones_like(y_true), -1), tf.expand_dims(y_true, axis = 1))
            score_diffs = tf.matmul(mat, _y_pred)
            score_diffs = tf.squeeze(score_diffs - _y_pred, -1) 
            loss_by_sample = tf.reduce_sum(tf.nn.sigmoid(tf.square(y_pred)), axis = -1) + \
                            tf.reduce_sum(tf.sigmoid(-score_diffs), axis = -1) + \
                            -tf.squeeze(tf.squeeze(tf.nn.sigmoid(tf.square(tf.matmul(tf.expand_dims(y_true, 1), _y_pred))), -1), -1)
            return tf.reduce_sum(loss_by_sample)

    '''
    데이터셋 Train, Test set
    '''
    df = pd.read_csv('../../3.홍홍홍_데이터 및 모델 세이브 파일/dataset/big_comp/main.csv', sep='\t')

    df = df.groupby("session").filter(lambda x: len(x) > 5)
    df_cold = df.groupby("session").filter(lambda x: len(x) <= 5)

    columns = ['session']
    for i in columns:
        globals()[f'encoder_{i}'] = LabelEncoder()
        df[i] = globals()[f'encoder_{i}'].fit_transform(df[i]) 
        

    df = df.sort_values(by = ['session', 'timestamp']).reset_index(drop = True)

    offsets = np.concatenate((np.zeros(1, dtype = np.int32), df.groupby('session').size().cumsum().values))

    dataset_train = SessionDataset(df.iloc[~df.index.isin(offsets[1:] - 1)]) 

    X_test = df.iloc[offsets[1:] - 2][['session', 'item']].sort_values('session').reset_index(drop = True)
    y_test = df.iloc[offsets[1:] - 1][['session', 'item']].sort_values('session').reset_index(drop = True)


    '''
    GRU4REC 학습
    '''
    g4r = Gru4Rec(n_classes = dataset_train.n_items)
    g4r.fit(dataset_train)

    final_states = np.empty(shape = (dataset_train.n_sessions, g4r.n_layers, g4r.n_hidden)) 
    final_states[:] = None
    done = [False] * dataset_train.n_sessions   

    stacks = [dataset_train.extract_session(i)[::-1] for i in range(g4r.batch_size)]
    next_session_id = g4r.batch_size
    batch_idx_to_session = np.arange(g4r.batch_size)  
    X = np.empty(shape = (g4r.batch_size, 1, g4r.n_classes))

    g4r.model.reset_states()  

    n_done = 0

    final_states = np.empty(shape = (dataset_train.n_sessions, g4r.n_layers, g4r.n_hidden))
    final_states[:] = None
    done = [False] * dataset_train.n_sessions  

    stacks = [dataset_train.extract_session(i)[::-1] for i in range(g4r.batch_size)]
    next_session_id = g4r.batch_size
    batch_idx_to_session = np.arange(g4r.batch_size)   
    X = np.empty(shape = (g4r.batch_size, 1, g4r.n_classes))

    g4r.model.reset_states()    

    n_done = 0

    while n_done < dataset_train.n_sessions:
        for i in range(g4r.batch_size):
            while len(stacks[i]) == 1:  
                if not done[batch_idx_to_session[i]]:
                    final_states[batch_idx_to_session[i], :] = np.array([layer.states[0][i, :] for layer in g4r.model.layers if g4r._is_GRU_layer(layer)])
                    done[batch_idx_to_session[i]] = True
                    n_done += 1
                    print(str(round(n_done/25884*100,1))+"% is completed.")
                    if n_done % 100 == 0:
                        print(f"Progress: {n_done} / {dataset_train.n_sessions}")
                if next_session_id >= dataset_train.n_sessions: 
                    next_session_id = 0
                stacks[i] = dataset_train.extract_session(next_session_id)[::-1]
                batch_idx_to_session[i] = next_session_id
                next_session_id += 1
                g4r._reset_hidden(i)  
            X[i, 0] = stacks[i].pop()

        _ = g4r.model.predict(X)   

    print("All final hidden states calculated")
    np.save('../../3.홍홍홍_데이터 및 모델 세이브 파일/dataset/big_comp/final_states.npy', final_states, allow_pickle = False)
    final_states = np.load('../dataset/big_comp/final_states.npy')

    g4r.model.reset_states()

    rem = dataset_train.n_sessions % g4r.batch_size
    if rem > 0:
        X_test = pd.concat((X_test, X_test[:(g4r.batch_size - rem)]), axis = 0)


    '''
    TEST 데이터 Accuracy 확인
    '''

    y_pred = np.empty(shape = (dataset_train.n_sessions, g4r.n_classes))
    y_pred[:] = None
    X = np.empty(shape = (g4r.batch_size, 1, g4r.n_classes))
    for batch_id in range(dataset_train.n_sessions // g4r.batch_size):
        X[:] = None
        for i in range(g4r.batch_size):
            X[i, :] = dataset_train.item_to_one_hot(X_test.iloc[batch_id * g4r.batch_size + i]['item'])
        nlg = 0
        for nl, layer in enumerate(g4r.model.layers):
            if g4r._is_GRU_layer(layer):
                g4r.model.layers[nl].reset_states(final_states[batch_id * g4r.batch_size : (batch_id + 1) * g4r.batch_size, nlg, :])
                nlg += 1

        y_pred[batch_id * g4r.batch_size : (batch_id + 1) * g4r.batch_size, :] = g4r.model.predict(X)[:g4r.batch_size]

    y_pred = tf.constant(y_pred[:dataset_train.n_sessions], dtype = tf.float32)

    y_true = np.empty(shape = (dataset_train.n_sessions, dataset_train.n_items))
    for i in range(y_true.shape[0]):
        y_true[i, :] = dataset_train.item_to_one_hot(y_test.item.values[i])
    y_true = tf.constant(y_true, dtype = tf.float32)

    acc       = (tf.reduce_sum(tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k = 1)) / y_true.shape[0]).numpy()
    top_10_acc = (tf.reduce_sum(tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k = 10)) / y_true.shape[0]).numpy()

    print("Accuracy = {}".format(acc))
    print("Top-10 accuracy = {}".format(top_10_acc))

    data_f = pd.DataFrame(y_pred)
    data_f.dropna(axis=0, inplace=True)

    columns = []
    for idx, val in enumerate(dataset_train.item_to_id):
        columns.append(val)
        
    data_f.columns = columns



    '''
    예측데이터 Rank.csv 생성
    '''
    df_rank = pd.DataFrame(columns=['cust', 'clac_hlv_nm', 'rank'], index=range(0, len(data_f)))

    count = 0 
    for i in tqdm(range(len(data_f))):
        res = data_f.iloc[i].sort_values(ascending=False)

        list_idx = []
        n = 0
        for idx, val in enumerate(res):
            if n < 15:
                list_idx.append(res.index[idx])
                n += 1
            else:
                break

        for j in range(len(list_idx)): 
            df_rank.loc[count, 'cust'] = i
            df_rank.loc[count, 'clac_hlv_nm'] = list_idx[j]
            df_rank.loc[count, 'rank'] = j + 1
            count += 1

    df_rank.to_csv('../../3.홍홍홍_데이터 및 모델 세이브 파일/dataset/big_comp/gru4rec_rank.csv', sep='\t', index=False)