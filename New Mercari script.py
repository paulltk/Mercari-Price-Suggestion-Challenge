import os
import nltk
from funcs import *

path = "C:\\Users\pault\OneDrive\Documenten\GitHub\input"
#path = "/home/afalbrecht/Documents/Leren en Beslissen/"
#path = "S:\OneDrive\Documenten\GitHub\input"

os.chdir(path)

train = pd.read_csv('train.tsv', delimiter='\t', encoding='utf-8')
train = train[train["price"] != 0] # Drops rows with price = 0
train.index = range(len(train))
print("train size:", train.shape)

start = time.time()
edit_data(train)
print(time.time() - start)
description_dict, category_dict, brand_dict = make_dictionaries(train)
print(time.time() - start)
sparse_matrix = make_sparse_matrix(data, description_dict, categories_dict, brand_dict)
# print(description_dict)
print(len(description_dict), len(category_dict), len(brand_dict))
