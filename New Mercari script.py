import numpy as np
import pandas as pd
import os
import time
from collections import Counter
import re
from sklearn.neural_network import MLPClassifier
import math

path = "C:\\Users\pault\OneDrive\Documenten\GitHub\input"
os.chdir(path)

train = pd.read_csv('train.tsv', delimiter='\t', encoding='utf-8')
print("train size:", train.shape)

#train = train.loc[0:100000]

def fill_missing_items(data):
    data["brand_name"].fillna("missing", inplace=True)
    data["item_description"].fillna("missing", inplace=True)
    data["category_name"].fillna("missing", inplace=True)

def list_to_dict(lst):
    return {k: v for v, k in enumerate(lst)}

def words_list(data):
    words = []
    data["item_description"] = data["item_description"].str.split()
    descr_list = data["item_description"].tolist()
    for sen in descr_list:
        for word in sen:
            if len(word) > 3:
                words.append(re.sub("[^0-9a-z]", "", word.lower()))
    count = Counter(words)
    return set([w for w in words if count[w] > 10])  # list with words

def edit_data_make_dicts(data):
    start = time.time()
    data["category_name"] = data["category_name"].str.split("/")
    print("split cat", time.time() - start)
    fill_missing_items(train)
    print("fill missing", time.time() - start)
    words = words_list(data)
    print("make words", time.time() - start)
    categories = []
    brand_names = []
    for cats in data["category_name"]:
        if cats == cats:
            categories += cats
    for brand in data["brand_name"]:
        brand_names.append(brand)
    print("append", time.time() - start)
    return list_to_dict(words), list_to_dict(list(set(categories))), list_to_dict(list(set(brand_names)))

def make_vector(row, description_dict, categories_dict, brand_dict):
    cat_v = [0] * len(categories_dict)
    for cat in row["category_name"]:
        try:
            cat_v[categories_dict[cat]] = 1
        except KeyError:
            pass
    brand_v = [0] * len(brand_dict)
    try:
        brand_v[brand_dict[row["brand_name"]]] = 1
    except KeyError:
        pass
    item_v = [0] * len(description_dict)
    for word in row["item_description"]:
        try:
            item_v[description_dict[word]] = 1
        except KeyError:
            pass
    cond_v = [0, 0, 0, 0, 0]
    cond_v[row["item_condition_id"] - 1] = 1
    vector = cat_v + brand_v + item_v + cond_v + [row["shipping"]]
    return vector

def get_price_list(data):
    lst = []
    for index, row in data.iterrows():
        lst.append(int(row["price"]))
    return np.array(lst)

def calc_score(prices, predicted_prices):
    summ = 0
    for price, pre_price in zip(prices, predicted_prices):
        summ += (math.log(int(pre_price)+1) - math.log(int(price)+1))**2
    return math.sqrt(summ / len(prices))

###############################################################################################
###############################################################################################

description_dict, categories_dict, brand_dict = edit_data_make_dicts(train)
print(len(description_dict), len(categories_dict), len(brand_dict))
vector_len = len(description_dict) + len(categories_dict) + len(brand_dict) + 6

data = train.loc[0:9999]

start = time.time()

print(start)

matrix = np.empty([10000, vector_len])
for index, row in data.iterrows():
    v = make_vector(row, description_dict, categories_dict, brand_dict)
    matrix[index] = v

prices = get_price_list(data)

print("made matrix", time.time() - start)

neuralnet = MLPClassifier(activation='relu', alpha=1e-04, batch_size='auto',
                        beta_1=0.9, beta_2=0.999, early_stopping=False,
                        epsilon=1e-08, hidden_layer_sizes=(15,), learning_rate='constant',
                        learning_rate_init=0.001, max_iter=20, momentum=0.9,
                        nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
                        solver='adam', tol=0.0001, validation_fraction=0.1, verbose=False,
                        warm_start=True)


#for i in range(5):


neuralnet.fit(matrix, prices)

print("fit matrix", time.time() - start)

predicted_prices = neuralnet.predict(matrix)

print("predict m", time.time() - start)

print("The score is:", calc_score(prices, predicted_prices))