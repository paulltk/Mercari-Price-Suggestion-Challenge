import numpy as np
import pandas as pd
import os
import statistics
import heapq
import re, string
import time
from collections import Counter
from sklearn.decomposition import PCA
from nltk.stem import WordNetLemmatizer

def lists_for_vector(data, thres=25):
    categories = []
    item_des = []
    brand_names = []
    s = time.time()
    for index, row in data.iterrows():
        categories += row["cat_list"]
        item_des += row["item_description"]
        brand_names.append(row["brand_name"])
    print("lists for vector time:", time.time() - s)
    des_count = Counter(item_des)
    item_des = [word for word in des_count if des_count[word] > thres]
    print(time.time() - s)
    return list(set(item_des)), list(set(categories)), list(set(brand_names))

def list_to_dict(lst):
    return {k: v for v, k in enumerate(lst)}

def instances_into_vectors(data, item_dict, cat_dict, brand_dict):
    """The vector is build out of the following elements: categories, brand name, item description,
    item condition and shipping. We have a list with all categories, a list with all brand names,
    same for item description. If categories is 1000 long, and brand name 2000, and the brand of an instance
    is on index 500, the boolean on the index 100+500 of the vector will be set to 1"""
    nodes = len(item_dict) + len(cat_dict) + len(brand_dict) + 6
    matrix = np.empty((data.shape[0], nodes))
    s = time.time()
    for index, row in data.iterrows():
        cat_v = [0] * len(cat_dict)
        for cat in row["cat_list"]:
            try: cat_v[cat_dict[cat]] =  1
            except KeyError: pass
        brand_v = [0] * len(brand_dict)
        try: brand_v[brand_dict[row["brand_name"]]] = 1
        except KeyError: pass
        item_v = [0] * len(item_dict)
        for word in row["item_description"]:
            try: item_v[item_dict[word]] = 1
            except KeyError: pass
        cond_v = [0, 0, 0, 0, 0]
        cond_v[row["item_condition_id"]-1] = 1
        vector = cat_v + brand_v + item_v + cond_v + [row["shipping"]]
        matrix[index] = vector
    print("instances to vector time:", time.time() - s)
    return matrix

def vector_to_csv(data, path):
    change_item_description(data)
    split_categories(data)
    fill_missing_items(data)
    item_des, categories, brand_names = lists_for_vector(data)
    with open(path, 'w') as vec_file:
        for i, cat in enumerate(categories):
            vec_file.write(str(i) + "," + cat + "\n")
        for i, brand in enumerate(brand_names):
            vec_file.write(str(i) + "," + brand + "\n")
        for i, word in enumerate(item_des):
            vec_file.write(str(i) + "," + word + "\n")
    pass

def get_price_list(data):
    lst = []
    for index, row in data.iterrows():
        lst.append(int(row["price"]))
    return lst

def calc_score(prices, predicted_prices):
    summ = 0
    for price, pre_price in zip(prices, predicted_prices):
        summ += (math.log(int(pre_price)+1) - math.log(int(price)+1))**2
    return math.sqrt(summ / len(prices))