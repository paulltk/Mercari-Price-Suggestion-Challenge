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
    data["brand_name"].fillna("missing", inplace=True)
    data["category_name"].fillna("missing", inplace=True)

def lower_string(text):
    return text.lower()


def edit_data(data):
    fill_missing_items(data)
    data["category_name"] = data["category_name"].apply(lower_string)
    data["category_name"] = data["category_name"].str.split("/")
    data["brand_name"] = data["brand_name"].apply(lower_string)


def make_dictionaries(data):
    categories = []
    for cats in data["category_name"]:
        if cats == cats:
            categories += cats
    categories = list(set(categories))
    category_dict = {k: v for v, k in enumerate(categories)}

    brand_names = []
    for brand in data["brand_name"]:
        brand_names.append(brand)
    count = Counter(brand_names)
    brand_list = set(list([b for b in brand_names if count[b] > 2]))
    brand_dict = {k: v for v, k in enumerate(brand_list)}
    return category_dict, brand_dict


def get_price_list(data):
    return np.array(data["price"].tolist())

def calc_score(prices, predicted_prices):
    summ = 0
    for price, pre_price in zip(prices, predicted_prices):
        summ += (math.log(int(pre_price)+1) - math.log(int(price)+1))**2
    return math.sqrt(summ / len(prices))


def make_sparse_matrix(data, categories_dict, brand_dict):
    sparse_matrix = sparse.lil_matrix((data.shape[0], len(categories_dict) + len(brand_dict) + 6), dtype=bool)
    cat_len, brand_len = len(categories_dict), len(brand_dict)

    categories_list = data["category_name"].tolist()
    for i, categories in enumerate(categories_list):
        for category in categories:
            if category in categories_dict:
                sparse_matrix[i, categories_dict[category]] = True

    brand_list = data["brand_name"].tolist()
    for i, brand in enumerate(brand_list):
        if brand in brand_dict:
            sparse_matrix[i, brand_dict[brand] + cat_len] = True
    condition_list = data["item_condition_id"].tolist()
    for i, condition in enumerate(condition_list):
        sparse_matrix[i, cat_len + brand_len + condition - 1] = True
    shipping_list = data["shipping"].tolist()
    for i, shipping in enumerate(shipping_list):
        sparse_matrix[i, cat_len + brand_len + 5] = shipping
    return sparse_matrix

def make_model(vec_len):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_dim=vec_len))
    model.add(Dense(96, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='msle', optimizer=optimizers.Adam())
#    model.compile(loss=RMSLE, optimizer=optimizers.Adam())
    return model

#################################################################################################
#################################################################################################
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

def no_itemdes(batchsize, amount):
    print("START NO ITEM DESCRIPTION NOW")
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
    categories_dict, brand_dict = make_dictionaries(train)
    print("made dictionaries", time.time() - start, " \n dict lengths:",
          len(categories_dict), len(brand_dict))
    sparse_matrix = make_sparse_matrix(train, categories_dict, brand_dict)
    print("made sparse matrix:", time.time() - start)
    prices = get_price_list(train)
    print("made prices", time.time() - start)
    vec_len = len(categories_dict) + len(brand_dict) + 6
    neuralnet = make_model(vec_len)
    print("neural net set up:", time.time() - start)

    return test_neuralnet(neuralnet, sparse_matrix, prices, batchsize, amount)