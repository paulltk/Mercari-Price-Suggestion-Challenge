import numpy as np
import pandas as pd
import statistics
import heapq
import re, string
import time
import math
from collections import Counter

from email import header
from sklearn.decomposition import PCA
from nltk.stem import WordNetLemmatizer

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

def calc_score(prices, predicted_prices):
    summ = 0
    for price, pre_price in zip(prices, predicted_prices):
        summ += (math.log(int(pre_price)+1) - math.log(int(price)+1))**2
    return math.sqrt(summ / len(prices))