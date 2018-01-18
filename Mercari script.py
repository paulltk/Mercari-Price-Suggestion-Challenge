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
#from vectorizenoitemdes import *
from vectorize import *

path = "C:\\Users\pault\OneDrive\Documenten\GitHub\input"
#path = "/home/afalbrecht/Documents/Leren en Beslissen/"
os.chdir(path)

train = pd.read_csv('train.tsv', delimiter='\t', encoding='utf-8')
print("train size:", train.shape)
test = pd.read_csv('test.tsv', delimiter='\t', encoding='utf-8')
print("test size:", test.shape)

data = train.loc[:100000]
valdata = train.loc[10000:20000]

start = time.time()
edit_data(train)
print("Edited data:", time.time() - start)
item_dict, cat_dict, brand_dict = get_dicts(train, 5)
cat_dict, brand_dict = get_dicts(data, 5)
print(len(item_dict), len(cat_dict), len(brand_dict))
print(time.time() - start)
#print(len(cat_dict), len(brand_dict))

data = train.loc[:10000]
valdata = train.loc[20000:30000]

print("Make dicts:", time.time() - start)
matrix = vectorize_data(data, item_dict, cat_dict, brand_dict)
#matrix = vectorize_data(data, cat_dict, brand_dict)

print("Made matrix:", time.time() - start, "matrix size:", matrix.shape)
# np.savetxt("test.csv", matrix, delimiter=",")
# print("Saved matrix:", time.time() - start)

prices = get_price_list(data)
print("price list:", time.time() - start, "price size:", len(prices))

pca_mat, pca = fit_PCA(matrix, 0.5)
print('PCA mat shape:', pca_mat.shape)

edit_data(valdata)
print("Edited valdata:", time.time() - start)
valmatrix = vectorize_data(valdata, item_dict, cat_dict, brand_dict)
#valmatrix = vectorize_data(valdata, cat_dict, brand_dict)

print("Made valmatrix:", time.time() - start, "valmatrix size:", valmatrix.shape)
valprices = get_price_list(valdata)
print("Valprice list:", time.time() - start, "valprice size:", len(valprices))

valmat = test_PCA(valmatrix, pca)
print('PCA valmat shape:', valmat.shape)

neuralnet = MLPClassifier(activation='relu', alpha=1e-04, batch_size='auto',
                        beta_1=0.9, beta_2=0.999, early_stopping=False,
                        epsilon=1e-08, hidden_layer_sizes=(15,), learning_rate='constant',
                        learning_rate_init=0.001, max_iter=200, momentum=0.9,
                        nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
                        solver='adam', tol=0.0001, validation_fraction=0.1, verbose=False,
                        warm_start=True)

neuralnet.fit(matrix, prices)

print('Model fitted', time.time() - start)
predicted_prices = neuralnet.predict(valmatrix)
print('Prices predicted', time.time() - start)

print("The score is:", calc_score(valprices, predicted_prices))

print("SECOND RUN")

data = train.loc[10000:20000]
start = time.time()
edit_data(data)
print("Edited data:", time.time() - start)
item_dict, cat_dict, brand_dict = get_dicts(data, 5)
#cat_dict, brand_dict = get_dicts(data, 5)
print(len(item_dict), len(cat_dict), len(brand_dict))
#print(len(cat_dict), len(brand_dict))
print("Make dicts:", time.time() - start)
matrix = vectorize_data(data, item_dict, cat_dict, brand_dict)
#matrix = vectorize_data(data, cat_dict, brand_dict)

print("Made matrix:", time.time() - start, "matrix size:", matrix.shape)

neuralnet.fit(matrix, prices)

neuralnet = fit_data(pca_mat, prices)
print('Model fitted', time.time() - start)
predicted_prices = neuralnet.predict(valmat)
print('Prices predicted', time.time() - start)

print("The score is:", calc_score(valprices, predicted_prices))