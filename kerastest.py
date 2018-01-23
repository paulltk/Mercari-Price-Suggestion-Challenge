import os
from funcs import *
from keras.models import Sequential

path = "C:\\Users\pault\OneDrive\Documenten\GitHub\input"
path = "/home/afalbrecht/Documents/Leren en Beslissen/"
#path = "S:\OneDrive\Documenten\GitHub\input"

os.chdir(path)

train = pd.read_csv('train.tsv', delimiter='\t', encoding='utf-8')
print("train size:", train.shape)

train = train[train["price"] != 0] # Drops rows with price = 0

#train = train.loc[:100000]
description_dict, categories_dict, brand_dict = edit_data_make_dicts(train)
# print(len(description_dict), len(categories_dict), len(brand_dict))

train = train.loc[:100000]
start = time.time()

def make_model():
    vector = cat_v + brand_v + item_v + cond_v + [row["shipping"]]
    item_desc = Input(shape=[len(item_desc)], name="item_desc")
    brand = Input(shape=[1], name="brand")
    category = Input(shape=[1], name="category")
    category_name = Input(shape=[X_train["category_name"].shape[1]],
                          name="category_name")
    item_condition = Input(shape=[1], name="item_condition")
    num_vars = Input(shape=[X_train["num_vars"].shape[1]], name="num_vars")


def index_des(row, dict, column_name):
    a = [dict[word] for word in row[column_name] if word in dict]
    return a

def brand(row, dict):
    return brand_dict[row["brand_name"]]

train["item_des"] = train.apply(lambda row: index_des(row, description_dict, "item_description"), axis=1)
train["cat_name"] = train.apply(lambda row: index_des(row, categories_dict, "category_name"), axis=1)
train["brand"] = train.apply(lambda row: brand(row, brand_dict), axis=1)

print(time.time() - start)

print(train.head(5))
# nn_multiple_times(train, 2, description_dict, categories_dict, brand_dict)
# train = train.loc[:10000]
# start = time.time()
# i = 0
matrix = []
for index, row in train.iterrows():
    v = make_vector(row, description_dict, categories_dict, brand_dict)
    matrix.append(v)



print(time.time() -start)