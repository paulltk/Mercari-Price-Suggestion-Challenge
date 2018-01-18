import numpy as np
import pandas as pd
import time
from collections import Counter

def get_dicts(data, thres=25):
    item_des, categories, brand_names = lists_for_vector(data, thres)
    item_dict, cat_dict, brand_dict = list_to_dict(item_des), list_to_dict(categories), list_to_dict(brand_names)
    return item_dict, cat_dict, brand_dict

#combines the functions below
def vectorize_data(data, item_dict, cat_dict, brand_dict):
    matrix = instances_into_vectors(data, item_dict, cat_dict, brand_dict)
    return matrix

#makes three lists, constisting of all catagories, brands and words in the item description we want to put in the NN.
def lists_for_vector(data, thres):
    categories = []
    item_des = []
    brand_names = []
    #s = time.time()
    for index, row in data.iterrows():
        categories += row["cat_list"]
        item_des += row["item_description"]
        brand_names.append(row["brand_name"])
    #print("lists for vector time:", time.time() - s)
    des_count = Counter(item_des)
    item_des = [word for word in des_count if des_count[word] > thres]
    #print(time.time() - s)
    return list(set(item_des)), list(set(categories)), list(set(brand_names))

# makes a dict from a list
def list_to_dict(lst):
    return {k: v for v, k in enumerate(lst)}

def instances_into_vectors(data, item_dict, cat_dict, brand_dict):
    """The vector is build out of the following elements: categories, brand name, item description,
    item condition and shipping. We have a list with all categories, a list with all brand names,
    same for item description. If categories is 1000 long, and brand name 2000, and the brand of an instance
    is on index 500, the boolean on the index 100+500 of the vector will be set to 1"""
    nodes = len(cat_dict) + len(brand_dict) + 6 + len(item_dict)
    matrix = np.empty((data.shape[0], nodes))
    #s = time.time()
    i = 0
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
        vector = cat_v + brand_v + cond_v + [row["shipping"]] + item_v
        matrix[i] = vector
        i += 1
    #print("instances to vector time:", time.time() - s)
    return matrix

#return a list with prices of products
def get_price_list(data):
    lst = []
    for index, row in data.iterrows():
        lst.append(int(row["price"]))
    return lst