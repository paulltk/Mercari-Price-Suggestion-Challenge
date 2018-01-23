import numpy as np
import pandas as pd
import os
import statistics
import heapq
import re, string
import time
from nltk.stem import WordNetLemmatizer

# Counts uniques in column
def count_uniques(data, column):
    total = set()
    nameless = []
    for val in data[column]:
        if val != None:
            total.update(val.split())
            nameless += val.split()
    return len(total), len(nameless)


# sorts a column and returns a dictionary with a list of indexes that have the same text/number in that column.
def sort_index_by_column(data, column):
    dct = {'nan': []}
    for index, row in data.iterrows():
        if row[column] == row[column]:
            if row[column] in dct:
                dct[row[column]].append(index)
            else:
                dct[row[column]] = [index]
        else:
            dct["nan"].append(index)
    return dct


# deletes all the keys that have a list with only one value
def delete_einzelgangers(dictionary):
    new_dict = {}
    for key in dictionary:
        if len(dictionary[key]) != 1:
            new_dict[key] = dictionary[key]
    return new_dict


# for every list, replace all the indexes with prices
def replace_index_with_price(dicti, data):
    new_dict = {}
    for key in dicti:
        lst = []
        for ind in dicti[key]:
            lst.append(data.get_value(ind, "price"))
        new_dict[key] = lst
    return new_dict


# calculates various things for every key
def mean_var_amount_per_dict_key(dicti):
    mean_dict = {}
    for key in dicti:
        mean = statistics.mean(dicti[key])
        variance = statistics.variance(dicti[key])
        amount = len(dicti[key])
        mean_dict[key] = [mean, variance, int(amount)]
    return mean_dict


# gets the amount of times a word is being used in the item description
def get_common_words(data):
    dct = {}
    for index, row in data.iterrows():
        for word in row["item_description"].split(" "):
            if word in dct:
                dct[word] += 1
            else:
                dct[word] = 1
    highest_keys = sorted(dct, key=dct.get, reverse=True)
    new_dct = {}
    for key in highest_keys:
        new_dct[key] = dct[key]
    return new_dct





















# def make_vector(row, description_dict, categories_dict, brand_dict):
#     cat_v = [0] * len(categories_dict)
#     for cat in row["category_name"]:
#         if cat in categories_dict:
#             cat_v[categories_dict[cat]] = 1
#     brand_v = [0] * len(brand_dict)
#     if row["brand_name"] in brand_dict:
#         brand_v[brand_dict[row["brand_name"]]] = 1
#     item_v = [0] * len(description_dict)
#     for word in row["item_description"]:
#         if word in description_dict:
#             item_v[description_dict[word]] = 1
#     cond_v = [0, 0, 0, 0, 0]
#     cond_v[row["item_condition_id"] - 1] = 1
#     vector = cat_v + brand_v + item_v + cond_v + [row["shipping"]]
#     return vector

def make_vector(row, description_dict, categories_dict, brand_dict):
    cat_v = [0] * len(categories_dict)
    for cat in row["cat_name"]:
        cat_v[cat] = 1
    brand_v = [0] * len(brand_dict)
    brand_v[row["brand"]] = 1
    item_v = [0] * len(description_dict)
    for ind in row["item_des"]:
        item_v[ind] = 1
    cond_v = [0, 0, 0, 0, 0]
    cond_v[row["item_condition_id"] - 1] = 1
    vector = cat_v + brand_v + item_v + cond_v + [row["shipping"]]
    return vector









def nn_multiple_times(train, i, description_dict, categories_dict, brand_dict):
    print("Start NN: 0")
    start = time.time()
    vector_len = len(description_dict) + len(categories_dict) + len(brand_dict) + 6

    neuralnet = MLPClassifier(activation='relu', alpha=1e-04, batch_size='auto',
                              beta_1=0.9, beta_2=0.999, early_stopping=False,
                              epsilon=1e-08, hidden_layer_sizes=(15,), learning_rate='constant',
                              learning_rate_init=0.001, max_iter=20, momentum=0.9,
                              nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
                              solver='adam', tol=0.0001, validation_fraction=0.1, verbose=False,
                              warm_start=False)

    valdata = train.loc[i * 10000:i * 10000 + 9999]
    valmatrix = np.empty([10000, vector_len])
    for index, row in valdata.iterrows():
        v = make_vector(row, description_dict, categories_dict, brand_dict)
        valmatrix[index-i*10000] = v
    print("Made valmatrix:", time.time() - start)

    for t in range(i):
        data = train.loc[t*10000:t*10000+9999]
        matrix = np.empty([10000, vector_len])
        for index, row in data.iterrows():
            v = make_vector(row, description_dict, categories_dict, brand_dict)
            matrix[index-t*10000] = v
        print("Made matrix", t, "in:", time.time() - start, matrix.shape)
        prices = get_price_list(data)
        neuralnet.fit(matrix, prices)
        print("Fitted matrix:", time.time() - start)

        predicted_prices = neuralnet.predict(valmatrix)
        print("The score is:", calc_score(prices, predicted_prices))