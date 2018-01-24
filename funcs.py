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


def edit_description_column(text):
    words = re.sub("[^a-z0-9 .]", "", text.lower())
    words = re.sub(r'(?<!\d)\.(?!\d)', '', words)
    return [word for word in words.split() if len(word) > 3]


def lower_string(text):
    return text.lower()


def edit_data(data):
    fill_missing_items(data)
    train["item_description"] = train["item_description"].apply(edit_description_column)
    train["category_name"] = train["category_name"].apply(lower_string)
    train["category_name"] = train["category_name"].str.split("/")
    train["brand_name"] = train["brand_name"].apply(lower_string)


def make_dictionaries(data):
    words_list = []
    description_list = data["item_description"].tolist()
    for sen in description_list:
        for word in sen:
            words_list.append(word)
    count = Counter(words_list)
    print("count", count["rmsh" ])
    des_list = list(set([w for w in words_list if count[w] > 20]))
    print(len(des_list))
    print("des list", time.time() - start)
    description_dict = {k: v for v, k in enumerate(des_list)}
    print("des dict", time.time() - start)

    categories = []
    for cats in data["category_name"]:
        if cats == cats:
            categories += cats
    print("cat list", time.time() - start)
    category_dict = {k: v for v, k in enumerate(categories)}
    print("cat dict", time.time() - start)

    brand_names = []
    for brand in data["brand_name"]:
        brand_names.append(brand)
    print("brand list", time.time() - start)
    count = Counter(brand_names)
    brand_list = [b for b in brand_names if count[b] > 2]
    brand_dict = {k: v for v, k in enumerate(brand_list)}
    print("brand dict", time.time() - start)
    return description_dict, category_dict, brand_dict


def get_price_list(data):
    lst = []
    for index, row in data.iterrows():
        lst.append(int(row["price"]))
    return np.array(lst)

# Replaces missing brand names with brand from item_description
def replace_missing_brands(data, brand_dict):
    for index, row in data.iterrows():
        if row["brand_name"] == 'missing':
            for word in row["item_description"]:
                if word in brand_dict:
                    data.at[data.index[index], "brand_name"] = word


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