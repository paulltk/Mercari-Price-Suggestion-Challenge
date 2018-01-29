import numpy as np
import pandas as pd
import time
from collections import Counter
import re
import math
from scipy import sparse
import os
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
from keras import backend as K

def fill_missing_items(data):
    data["item_description"].fillna("missing", inplace=True)


def edit_description_column(text):
    words = re.sub("[^a-z0-9 .]", "", text.lower())
    words = re.sub(r'(?<!\d)\.(?!\d)', '', words)
    return [word for word in words.split() if len(word) > 3]


def lower_string(text):
    return text.lower()


def edit_data(data):
    fill_missing_items(data)
    data["item_description"] = data["item_description"].apply(edit_description_column)


def make_dictionaries(data):
    words_list = []
    description_list = data["item_description"].tolist()
    for sen in description_list:
        for word in sen:
            words_list.append(word)
    count = Counter(words_list)
    des_list = list(set([w for w in words_list if count[w] > 20]))
    description_dict = {k: v for v, k in enumerate(des_list)}

    return description_dict


def get_price_list(data):
    return np.array(data["price"].tolist())


def calc_score(prices, predicted_prices):
    summ = 0
    for price, pre_price in zip(prices, predicted_prices):
        summ += (math.log(int(pre_price)+1) - math.log(int(price)+1))**2
    return math.sqrt(summ / len(prices))


def make_sparse_matrix(data, description_dict):
    sparse_matrix = sparse.lil_matrix((data.shape[0], len(description_dict)), dtype=bool)
    des_len = len(description_dict)

    descriptions_list = data["item_description"].tolist()
    for i, sen in enumerate(descriptions_list):
        for word in sen:
            if word in description_dict:
                sparse_matrix[i, description_dict[word]] = True
    return sparse_matrix

def make_model(vec_len):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_dim=vec_len))
    model.add(Dense(96, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='msle', optimizer=optimizers.Adam())
#    model.compile(loss=RMSLE, optimizer=optimizers.Adam())
    return model

########################################################################################
########################################################################################
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
    return calc_score(valbatchprices, predicted_prices), predicted_prices


def only_itemdes(batchsize, amount):
    print("START ONLY ITEM DESCRIPTION NOW")
    path = "C:\\Users\pault\OneDrive\Documenten\GitHub\input"
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
    description_dict = make_dictionaries(train)
    print("made dictionaries", time.time() - start, " \n dict lengths:",
          len(description_dict))
    sparse_matrix = make_sparse_matrix(train, description_dict)
    print("made sparse matrix:", time.time() - start)
    prices = get_price_list(train)
    print("made prices", time.time() - start)
    vec_len = len(description_dict)
    neuralnet = make_model(vec_len)
    print("neural net set up:", time.time() - start)
    return test_neuralnet(neuralnet, sparse_matrix, prices, batchsize, amount)