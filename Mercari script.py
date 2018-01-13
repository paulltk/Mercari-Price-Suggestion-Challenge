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
path = "/home/afalbrecht/Documents/Leren en Beslissen/" #add your path here, dit is een test
os.chdir(path)

train = pd.read_csv('train.tsv', delimiter='\t', encoding='utf-8')
print("train size:", train.shape)

test = pd.read_csv('test.tsv', delimiter='\t', encoding='utf-8')
print("test size:", test.shape)

data = train.loc[:11000]
#val = train.loc[0:1000]
edit_data(data)
#edit_data(val)
item_des, categories, brand_names = lists_for_vector(data)
item_dict, cat_dict, brand_dict = list_to_dict(item_des), list_to_dict(categories), list_to_dict(brand_names)

matrix = instances_into_vectors(data, item_dict, cat_dict, brand_dict)
val_mat = matrix[10000:11000]
matrix = matrix[:10000]

print(matrix.shape)
print(val_mat.shape)

def get_price_list(data):
    lst = []
    for index, row in data.iterrows():
        lst.append(int(row["price"]))
    return lst


prices = get_price_list(data[:10000])
val_prices = get_price_list(data[10000:11000])

clf = MLPClassifier(activation='relu', alpha=1e-04, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(15,), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='adam', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)

start = time.time()
clf.fit(matrix, prices)
print('Model fitted', time.time() - start)
predicted_prices = clf.predict(val_mat)
print('Prices predicted', time.time() - start)
a = 0
for price, pre_price in zip(val_prices, predicted_prices):
    if a <100:
        print(price, pre_price)
        a += 1

def calc_score(prices, predicted_prices):
    summ = 0
    for price, pre_price in zip(prices, predicted_prices):
        summ += (math.log(int(pre_price)+1) - math.log(int(price)+1))**2
    return math.sqrt(summ / len(prices))

print(calc_score(val_prices, predicted_prices))