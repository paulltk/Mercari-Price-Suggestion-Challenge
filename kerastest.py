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
    model.compile(loss="msle", optimizer=optimizers.Adam())
    return model

########################################################################################
########################################################################################

path = "C:\\Users\pault\OneDrive\Documenten\GitHub\input"
# path = "/home/afalbrecht/Documents/Leren en Beslissen/"
#path = "S:\OneDrive\Documenten\GitHub\input"
path = "/home/lisa/Documents/leren_beslissen/Merri/"
path = "/Users/falksinke/LocalDocs/mercari_project"


os.chdir(path)

start = time.time()

train = pd.read_csv('train.tsv', delimiter='\t', encoding='utf-8')
train = train[train["price"] < 300]
train = train[train["price"] != 0]  # Drops rows with price = 0
train.index = range(len(train))
# train = train.loc[0:100000]


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

def test_neuralnet(neuralnet, sparse_matrix, prices, batchsize, amount_batches=1):
    valbatch = sparse_matrix[amount_batches * batchsize: amount_batches * batchsize + 10000].todense()
    valbatchprices = prices[amount_batches * batchsize: amount_batches * batchsize + 10000]
    for t in range(amount_batches):
        print("this is batch", t)
        batch = sparse_matrix[t * batchsize:t * batchsize + batchsize].todense()
        batchprices = prices[t * batchsize:t * batchsize + batchsize]
        neuralnet.fit(batch, batchprices)
        predicted_prices = neuralnet.predict(valbatch)
        print("The score is:", calc_score(valbatchprices, predicted_prices))
        return predicted_prices
45

test_neuralnet(neuralnet, sparse_matrix, prices, 1000, amount_batches=100)

predprice = test_neuralnet(neuralnet, sparse_matrix, prices, 10000, amount_batches=1)
predprice.astype(float)
index_vector = [x for x in range(len(predprice))]
result = np.column_stack((index_vector, predprice))
np.savetxt("submission.csv", result, delimiter=",")
