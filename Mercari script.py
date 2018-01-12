import numpy as np
import pandas as pd
import os
import statistics
import heapq
import re, string
import time
import nltk
from mercarifunc import *

path = "C:\\Users\pault\OneDrive\Documenten\GitHub\input" # pauls path
path = "/home/afalbrecht/Documents/Leren en Beslissen/" #add your path here, dit is een test
os.chdir(path)

train = pd.read_csv('train.tsv', delimiter='\t', encoding='utf-8')
print("train size:", train.shape)

test = pd.read_csv('test.tsv', delimiter='\t', encoding='utf-8')
print("test size:", test.shape)

sample = train.loc[0:1000]


sample1 = train.loc[0:10000]
change_item_description(sample1)
split_categories(sample1)
fill_missing_brand_names(sample1)
matrix = instances_into_vectors(sample1)

print(matrix)