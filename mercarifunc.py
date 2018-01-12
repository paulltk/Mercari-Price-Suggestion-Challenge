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


# Replace NaN in brand_name column
def replace_nan_brand(data):
    data['brand_name'] = data['brand_name'].fillna('missing')
    return data


def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)


# every item (except a few NaN's) have three categories. Here we split those in three different columns.
def split_categories(data):
    data["cat_list"] = ""
    for index, row in data.iterrows():
        if row["category_name"] == row["category_name"]:
            catlist = [row["category_name"].split("/")[0],
                       row["category_name"].split("/")[1],
                       row["category_name"].split("/")[2]]
            # print(index, catlist)
            data.set_value(index, "cat_list", catlist)
        # print(row)
    return data


# replaces the item description with
def change_item_description(data):
    wnlm = WordNetLemmatizer()
    stopwords = set("""rm i me my myself we our ours ourselves you your yours yourself yourselves he him 
            his himself she her hers herself it its itself they them their theirs themselves what which who whom 
            this that these those am is are was were be been being have has had having do does did doing a an the 
            and but if or because as until while of at by for with about against between into through during before 
            after above below to from up down in out on off over under again further then once here there when where 
            why how all any both each few more most other some such no nor not only own same so than too very s t can 
            will just don should now""".split(" "))
    for index, row in data.iterrows():
        if isinstance(row["item_description"], list):
            continue
        des = row["item_description"]
        if not isinstance(des, str):
            print(des)
        des = des.lower()
        regex = re.compile('[' + re.escape(string.punctuation) + '\\r\\t\\n ]')
        txt = regex.sub(" ", des)
        words = [wnlm.lemmatize(w) for w in txt.split(" ") if w not in stopwords]
        data.set_value(index, "item_description", words)

def fill_missing_brand_names(data):
    data["brand_name"].fillna("missing", inplace=True)

def fill_missing_items(data):
    data["brand_name"].fillna("missing", inplace=True)
    data["item_description"].fillna("missing", inplace=True)

def lists_for_vector(data, thres=3):
    categories = []
    item_des = []
    brand_names = []
    start = time.time()
    print(time.time() - start)
    for index, row in data.iterrows():
        #print(row["first_cat"])
        categories += row["cat_list"]
        if isinstance(row["item_description"], str):
            print(index)
        item_des += row["item_description"]
        brand_names.append(row["brand_name"])
    print(time.time() - start)
    des_count = Counter(item_des)
    item_des = [word for word in des_count if des_count[word] > thres]
    print(time.time() - start)
    return list(set(item_des)), list(set(categories)), list(set(brand_names))

def list_to_dict(lst):
    return {k: v for v, k in enumerate(lst)}

def instances_into_vectors(data):
    """The vector is build out of the following elements: categories, brand name, item description,
    item condition and shipping. We have a list with all categories, a list with all brand names,
    same for item description. If categories is 1000 long, and brand name 2000, and the brand of an instance
    is on index 500, the boolean on the index 100+500 of the vector will be set to 1"""
    print(time.time())
    item_des, categories, brand_names = lists_for_vector(data)
    item_dict, cat_dict, brand_dict = list_to_dict(item_des), list_to_dict(categories), list_to_dict(brand_names)
    print(time.time())
    nodes = len(item_des) + len(categories) + len(brand_names) + 6
    #print(nodes)
    #print(data.shape[0])
    matrix = np.empty((data.shape[0], nodes))
    #print(matrix.shape)
    #print(item_des)
    #print(categories)
    for index, row in data.iterrows():
        cat_v = [0] * len(categories)
        for cat in row["cat_list"]:
            try: cat_v[cat_dict[cat]] =  1
            except KeyError: pass
        brand_v = [0] * len(brand_names)
        try: brand_v[brand_dict[row["brand_name"]]] = 1
        except KeyError: pass
        item_v = [0] * len(item_des)
        for word in row["item_description"]:
            try: item_v[item_dict[word]] = 1
            except KeyError: pass
        #cat_v = [1 if cat in row["cat_list"] else 0 for cat in categories]
        #brand_v = [1 if brand in row["brand_name"] else 0 for brand in brand_names]
        #item_v = [1 if word in row["item_description"] else 0 for word in item_des]
        cond_v = [0, 0, 0, 0, 0]
        cond_v[row["item_condition_id"]-1] = 1
        vector = cat_v + brand_v + item_v + cond_v + [row["shipping"]]
        matrix[index] = vector
    print(time.time())
    return matrix

def vector_to_csv(data, path):
    change_item_description(data)
    split_categories(data)
    fill_missing_brand_names(data)
    item_des, categories, brand_names = lists_for_vector(data)
    with open(path, 'w') as vec_file:
        for i, cat in enumerate(categories):
            vec_file.write(str(i) + "," + cat + "\n")
        for i, brand in enumerate(brand_names):
            vec_file.write(str(i) + "," + brand + "\n")
        for i, word in enumerate(item_des):
            vec_file.write(str(i) + "," + word + "\n")
    pass
