import os
from funcs import *
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers

def make_model(vec_len):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_dim=vec_len))
    model.add(Dense(96, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss="mse", optimizer=optimizers.Adam())
    return model

########################################################################################
########################################################################################

path = "C:\\Users\pault\OneDrive\Documenten\GitHub\input"
# path = "/home/afalbrecht/Documents/Leren en Beslissen/"
#path = "S:\OneDrive\Documenten\GitHub\input"
path = "/home/lisa/Documents/leren_beslissen/Merri/"


os.chdir(path)

start = time.time()

train = pd.read_csv('train.tsv', delimiter='\t', encoding='utf-8')
#train = train.loc[0:100000]


remove_zero_prices(train)
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
neuralnet = make_model(vec_len)
print("neural net set up:", time.time() - start)

valmat = sparse_matrix[9000:10000].todense()
valprices = get_price_list(train[9000:10000])
print("valmat shape:", valmat.shape)
print("val prices shape:", valprices.shape)

def test_neuralnet(neuralnet, sparse_matrix, prices, batchsize, amount_batches=1):
    for t in range(amount_batches):
        batch = sparse_matrix[t * batchsize:t * batchsize + batchsize].todense()
        batchprices = prices[t * batchsize:t * batchsize + batchsize]
        neuralnet.fit(batch, batchprices)
    valbatch = sparse_matrix[amount_batches * batchsize : amount_batches * batchsize + batchsize].todense()
    valbatchprices = prices[amount_batches * batchsize : amount_batches * batchsize + batchsize]
    predicted_prices = neuralnet.predict(valbatch)
    print("The score is:", calc_score(valbatchprices, predicted_prices))

test_neuralnet(neuralnet, sparse_matrix, prices, 10000, amount_batches=10)


