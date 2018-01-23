import numpy as np
import pandas as pd
import time
from collections import Counter
import re
from sklearn.neural_network import MLPClassifier
import math
from scipy import sparse

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
    return list(set([w for w in words if count[w] > 10]))  # list with words


def edit_data_make_dicts(data):
    start = time.time()
    data["category_name"] = data["category_name"].str.split("/")
    print("split cat", time.time() - start)
    fill_missing_items(data)
    print("fill missing", time.time() - start)
    words = words_list(data)
    words_dict = list_to_dict(words)
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

def make_sparse_matrix(data, description_dict, categories_dict, brand_dict):
    sparse_matrix = sparse.lil_matrix((data.shape[0], len(description_dict) + len(categories_dict) + len(brand_dict) + 6), dtype=bool)
    print(sparse_matrix.shape)
    des_len = len(description_dict)
    cat_len = len(categories_dict)
    brand_len = len(brand_dict)

    descriptions_list = data["item_description"].tolist()
    for i, sen in enumerate(descriptions_list):
        for word in sen:
            if word in description_dict:
                sparse_matrix[i, description_dict[word]] = True

    categories_list = data["category_name"].tolist()
    for i, categories in enumerate(categories_list):
        for category in categories:
            if category in categories_dict:
                sparse_matrix[i, categories_dict[category] + des_len] = True

    brand_list = data["brand_name"].tolist()
    for i, brand in enumerate(brand_list):
        if brand in brand_dict:
            sparse_matrix[i, brand_dict[brand] + des_len + cat_len] = True
    condition_list = data["item_condition_id"].tolist()
    for i, condition in enumerate(condition_list):
        sparse_matrix[i, des_len + cat_len + brand_len + condition - 1] = True
    shipping_list = data["shipping"].tolist()
    for i, shipping in enumerate(shipping_list):
        sparse_matrix[i, des_len + cat_len + brand_len + 5] = shipping
    return sparse_matrix