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
from otherfunc import *
from preprocessing import *
from vectorize import *
from neuralnetwork import *

path = "C:\\Users\pault\OneDrive\Documenten\GitHub\input"
#path = "/home/afalbrecht/Documents/Leren en Beslissen/"
os.chdir(path)

train = pd.read_csv('train.tsv', delimiter='\t', encoding='utf-8')
print("train size:", train.shape)
test = pd.read_csv('test.tsv', delimiter='\t', encoding='utf-8')
print("test size:", test.shape)

data = train.loc[:50000]
valdata = train.loc[50000:100000]

start = time.time()
edit_data(data)
print("Edited data:", time.time() - start)
item_dict, cat_dict, brand_dict = get_dicts(data)
print("Make dicts:", time.time() - start)
matrix = vectorize_data(data, item_dict, cat_dict, brand_dict)
print("Made matrix:", time.time() - start, "matrix size:", matrix.shape)
prices = get_price_list(data)
print("price list:", time.time() - start, "price size:", len(prices))

edit_data(valdata)
print("Edited valdata:", time.time() - start)
valmatrix = vectorize_data(valdata, item_dict, cat_dict, brand_dict)
print("Made valmatrix:", time.time() - start, "valmatrix size:", valmatrix.shape)
valprices = get_price_list(valdata)
print("Valprice list:", time.time() - start, "valprice size:", len(valprices))


neuralnet = fit_data(matrix, prices)
print('Model fitted', time.time() - start)
predicted_prices = neuralnet.predict(valmatrix)
print('Prices predicted', time.time() - start)

# a = 0
# i = 0
# for price, pre_price in zip(prices, predicted_prices):
#     if a <300:
#         print(i, price, pre_price)
#         print(data.get_value(i, "item_description"))
#         a += 1
#         i += 1

print("The score is:", calc_score(valprices, predicted_prices))