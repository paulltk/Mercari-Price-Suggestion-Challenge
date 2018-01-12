import numpy as np
import pandas as pd
import os
import statistics
import heapq
import re, string
import time
from nltk.stem import WordNetLemmatizer

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

# every item (except a few NaN's) have three categories. Here we make a list with those three categories.
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
        regex = re.compile('[' + re.escape(string.punctuation) + '\\r\\t\\n]')
        txt = regex.sub(" ", des)
        words = [wnlm.lemmatize(w) for w in txt.split(" ") if w not in stopwords]
        data.set_value(index, "item_description", words)

def fill_missing_items(data):
    data["brand_name"].fillna("missing", inplace=True)
    data["item_description"].fillna("missing", inplace=True)

def lists_for_vector(data):
    categories = []
    item_des = []
    brand_names = []
    s = time.time()
    for index, row in data.iterrows():
        categories += row["cat_list"]
        item_des += row["item_description"]
        brand_names.append(row["brand_name"])
    print("lists for vector time:", time.time() - s)
    return list(set(item_des)), list(set(categories)), list(set(brand_names))

def list_to_dict(lst):
    return {k: v for v, k in enumerate(lst)}

def instances_into_vectors(data, item_dict, cat_dict, brand_dict):
    """The vector is build out of the following elements: categories, brand name, item description,
    item condition and shipping. We have a list with all categories, a list with all brand names,
    same for item description. If categories is 1000 long, and brand name 2000, and the brand of an instance
    is on index 500, the boolean on the index 100+500 of the vector will be set to 1"""
    nodes = len(item_des) + len(categories) + len(brand_names) + 6
    matrix = np.empty((data.shape[0], nodes))
    s = time.time()
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
        cond_v = [0, 0, 0, 0, 0]
        cond_v[row["item_condition_id"]-1] = 1
        vector = cat_v + brand_v + item_v + cond_v + [row["shipping"]]
        matrix[index] = vector
    print("instances to vector time:", time.time() -s)
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
