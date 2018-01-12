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
