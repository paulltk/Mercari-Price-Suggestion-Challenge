import os

from funcs import *

path = "C:\\Users\pault\OneDrive\Documenten\GitHub\input"
#path = "S:\OneDrive\Documenten\GitHub\input"

os.chdir(path)

train = pd.read_csv('train.tsv', delimiter='\t', encoding='utf-8')
print("train size:", train.shape)

description_dict, categories_dict, brand_dict = edit_data_make_dicts(train)

start = time.time()

sparse_matrix = make_sparse_matrix(train, description_dict, categories_dict, brand_dict)

print(time.time() - start)

b = sparse_matrix.todense()[0:9999]

print(time.time() - start)
