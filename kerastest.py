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
test = pd.read_csv('test.tsv', delimiter='\t', encoding='utf-8')
train = train[train["price"] < 300]
train = train[train["price"] != 0]  # Drops rows with price = 0
train.index = range(len(train))
train = train.loc[0:1000]


sparse_matrix, prices, vec_len, dicts = preprocess_training(train, start)
neuralnet = make_model(vec_len)
print("neural net set up:", time.time() - start)


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
        neuralnet.fit(batch, batchprices)
        #predicted_prices = neuralnet.predict(valbatch)
        print("The score is:", calc_score(valbatchprices, predicted_prices))
#test_neuralnet(neuralnet, sparse_matrix, prices, 1000, amount_batches=100)



#predprice = test_neuralnet(neuralnet, sparse_matrix, prices, 10000, amount_batches=1)
#predprice.astype(float)
#index_vector = [x for x in range(len(predprice))]
#result = np.column_stack((index_vector, predprice))
#result.to_csv("submiss.csv", result, delimiter=",")
