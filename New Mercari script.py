import os

from funcs import *

path = "C:\\Users\pault\OneDrive\Documenten\GitHub\input"
path = "S:\OneDrive\Documenten\GitHub\input"

os.chdir(path)

train = pd.read_csv('train.tsv', delimiter='\t', encoding='utf-8')
print("train size:", train.shape)
train = train.loc[:100000]

description_dict, categories_dict, brand_dict = edit_data_make_dicts(train)
# print(len(description_dict), len(categories_dict), len(brand_dict))

print(train.head(5))
# nn_multiple_times(train, 2, description_dict, categories_dict, brand_dict)
train = train.loc[:10000]
start = time.time()
i = 0
for index, row in train.iterrows():
    print(row["item_description"])
    #v = make_vector(row, description_dict, categories_dict, brand_dict)
    

print(time.time() -start)