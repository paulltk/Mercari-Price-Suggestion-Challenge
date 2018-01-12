import numpy as np
import pandas as pd
import os
import statistics
import heapq
import re, string
import time
import math
from collections import Counter
from sklearn.decomposition import PCA
from nltk.stem import WordNetLemmatizer
from sklearn.neural_network import MLPClassifier
from mercarifunc import *

path = "C:\\Users\pault\OneDrive\Documenten\GitHub\input" # pauls path
#path = "/home/afalbrecht/Documents/Leren en Beslissen/" #add your path here, dit is een test
os.chdir(path)

train = pd.read_csv('train.tsv', delimiter='\t', encoding='utf-8')
print("train size:", train.shape)

test = pd.read_csv('test.tsv', delimiter='\t', encoding='utf-8')
print("test size:", test.shape)

data = train.loc[0:10000]
edit_data(data)
item_des, categories, brand_names = lists_for_vector(data)
item_dict, cat_dict, brand_dict = list_to_dict(item_des), list_to_dict(categories), list_to_dict(brand_names)

matrix = instances_into_vectors(data, item_dict, cat_dict, brand_dict)

print(matrix.shape)

def get_price_list(data):
    lst = []
    for index, row in data.iterrows():
        lst.append(int(row["price"]))
    return lst


prices = get_price_list(data)

clf = MLPClassifier(activation='relu', alpha=1e-04, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(15,), learning_rate='constant',
       learning_rate_init=0.001, max_iter=100, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)

clf.fit(matrix, prices)

predicted_prices = clf.predict(matrix)

a = 0
for price, pre_price in zip(prices, predicted_prices):
    if a <100:
        print(price, pre_price)
        a += 1

def calc_score(prices, predicted_prices):
    summ = 0
    for price, pre_price in zip(prices, predicted_prices):
        summ += (math.log(int(pre_price)+1) - math.log(int(price)+1))^2
    return summ / len(prices)

print(calc_score(prices, predicted_prices))