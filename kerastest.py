import os
from funcs import *
from sklearn.linear_model import Ridge
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Dense, BatchNormalization, \
    Activation, concatenate, GRU, Embedding, Flatten
from keras.preprocessing.sequence import pad_sequences
from keras import optimizers
from keras import backend as K

MAX_ITEM_DESC_SEQ = 60

def desc_seq(data, description_dict):
    desc_list = []
    descriptions_list = data["item_description"].tolist()
    for i, sen in enumerate(descriptions_list):
        desc = []
        for word in sen:
            if word in description_dict and len(desc):
                desc.append(description_dict[word])
        desc_list.append(desc)
    desc_seq = pad_sequences(desc_list, maxlen=MAX_ITEM_DESC_SEQ)
    return desc_seq

def embed_model(desc_len, vec_len):
    desc_in = Input(shape=MAX_ITEM_DESC_SEQ, name="desc_in")
    emb_item_desc = Embedding(len(description_dict), MAX_ITEM_DESC_SEQ)(desc_in)
    desc_layer = GRU(32) (emb_item_desc)
    desc_out = Dense(1, activation='linear', name='desc_out')(desc_layer)
    second_input = Input(shape=(vec_len - len(description_dict),), name='second_input')
    main_l = Dropout(0.3)(Dense(512, activation='relu') (second_input))
    main_l = Dropout(0.2)(Dense(96, activation='relu') (main_l))
    main_l = concatenate([desc_out, main_l])
    main_l = Dropout(0.2)(Dense(64, activation='relu') (main_l))
    output = Dense(1, activation="linear", name='output') (main_l)
    model = Model(inputs=[desc_in, second_input], outputs=[output, desc_out])
    optimizer = optimizers.Adam()
    model.compile(loss="msle",
                  optimizer=optimizer, loss_weights=[1., 0.2])
    return model



def make_model(vec_len):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_dim=vec_len))
    model.add(Dense(96, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss="msle", optimizer=optimizers.Adam())
    return model

def RMSLE(y_true, y_pred):
    return K.sqrt(K.mean(K.square(K.log(y_pred + 1) - K.log(y_true + 1)), axis=-1))

########################################################################################
########################################################################################

path = "C:\\Users\pault\OneDrive\Documenten\GitHub\input"
path = "/home/afalbrecht/Documents/Leren en Beslissen/"
#path = "S:\OneDrive\Documenten\GitHub\input"
#path = "/home/lisa/Documents/leren_beslissen/Merri/"


os.chdir(path)

start = time.time()

train = pd.read_csv('train.tsv', delimiter='\t', encoding='utf-8')
#test = pd.read_csv('test.tsv', delimiter='\t', encoding='utf-8')
train = train[train["price"] < 300]
train = train[train["price"] != 0]  # Drops rows with price = 0
train.index = range(len(train))
train = train.loc[0:100000]


print("train size:", train.shape)
edit_data(train)
print("edited all the data", time.time() - start)
description_dict, categories_dict, brand_dict = make_dictionaries(train)
print("made dictionaries", time.time() - start, " \n dict lengths:",
      len(description_dict), len(categories_dict), len(brand_dict))
sparse_matrix = make_sparse_matrix(train, description_dict, categories_dict, brand_dict)
print("made sparse matrix:", time.time() - start)
prices = get_price_list(train)
print("made prices", time.time() - start)
vec_len = len(description_dict) + len(categories_dict) + len(brand_dict) + 6
desc_seq = desc_seq(train, description_dict)
#neuralnet = make_model(vec_len)
neuralnet = embed_model(len(description_dict), vec_len)
# sparse_matrix, prices, vec_len, dicts = preprocess_training(train, start)
# neuralnet = make_model(vec_len)
print("neural net set up:", time.time() - start)
#ridge = Ridge(solver="sag", fit_intercept=False, alpha = 1.5, random_state=666)


batches = iter_minibatches(sparse_matrix, prices, chunksize=50)
for X_chunk, y_chunk in batches:
   if len(X_chunk) != 0:
       neuralnet.train_on_batch(X_chunk, y_chunk)
print("fit model")

sparse_test_matrix = preprocess_test(test, dicts, start)
print("sparse test matrix made")
neuralnet.predict(sparse_test_matrix.todense())

def test_neuralnet(neuralnet, sparse_matrix, prices, batchsize, amount_batches=1):
    valbatch = sparse_matrix[amount_batches * batchsize: amount_batches * batchsize + 10000].todense()
    valbatchprices = prices[amount_batches * batchsize: amount_batches * batchsize + 10000]
    for t in range(amount_batches):
        print("this is batch", t)
        batch = sparse_matrix[t * batchsize:t * batchsize + batchsize].todense()
        batchprices = prices[t * batchsize:t * batchsize + batchsize]
        neuralnet.fit({'desc_input': desc, 'aux_input': additional_data},
                  {'main_output': labels, 'aux_output': labels},
                  epochs=50, batch_size=32)
        #neuralnet.fit(batch, batchprices)
        predicted_prices = model.predict(X=valbatch)
        # predicted_prices = neuralnet.predict(valbatch)
        print("The score is:", calc_score(valbatchprices, predicted_prices))
#test_neuralnet(neuralnet, sparse_matrix, prices, 1000, amount_batches=100)

test_neuralnet(neuralnet, sparse_matrix, prices, 10000, amount_batches=1)


#predprice = test_neuralnet(neuralnet, sparse_matrix, prices, 10000, amount_batches=1)
#predprice.astype(float)
#index_vector = [x for x in range(len(predprice))]
#result = np.column_stack((index_vector, predprice))
#result.to_csv("submiss.csv", result, delimiter=",")
