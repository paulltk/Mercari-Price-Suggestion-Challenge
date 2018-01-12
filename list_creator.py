import numpy as np
import pandas as pd
import os
import statistics
import heapq
import re, string
import time
from nltk.stem import WordNetLemmatizer
import csv
from mercarifunc import *

path = "/home/afalbrecht/Documents/Leren en Beslissen/" #add your path here, dit is een test
os.chdir(path)

train = pd.read_csv('train.tsv', delimiter='\t', encoding='utf-8')
print("train size:", train.shape)

test = pd.read_csv('test.tsv', delimiter='\t', encoding='utf-8')
print("test size:", test.shape)

data = train

start = time.time()
fill_missing_items(data)
change_item_description(data)
print(time.time() - start)
split_categories(data)
print(time.time() - start)
item_des, categories, brand_names = lists_for_vector(data, thres=25)
print(len(item_des))


print(time.time() - start)
with open('vector.csv', 'w') as vec_file:
    for i, cat in enumerate(categories):
        vec_file.write(str(i) + ", " + cat + "\n")
    print(time.time() - start)
    for i, brand in enumerate(brand_names):
        vec_file.write(str(i) + ", " + brand + "\n")
    print(time.time() - start)
    for i, word in enumerate(item_des):
        vec_file.write(str(i) + ", " + word + "\n")
    print(time.time() - start)

