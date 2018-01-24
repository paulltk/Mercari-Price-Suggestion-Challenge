import os
from funcs import *
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers

def make_model(vec_len):
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=vec_len))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss="mse", optimizer=optimizers.Adam())
    return model



def index_des(row, dict, column_name):
    a = [dict[word] for word in row[column_name] if word in dict]
    return a

def brand(row, dict):
    return brand_dict[row["brand_name"]]

def get_rows(mat, price, rng):
    return mat[rng].todense(), price[rng]

def iter_minibatches(mat, price, chunksize):
    # Provide chunks one by one
    chunkstartmarker = 0
    while chunkstartmarker < mat.shape[0]:
        if (mat.shape[0] - chunkstartmarker) < chunksize:
            chunksize = mat.shape[0] - chunkstartmarker
        chunkrows = range(chunkstartmarker,chunkstartmarker+chunksize)
        X_chunk, y_chunk = get_rows(mat, price, chunkrows)
        yield X_chunk, y_chunk
        chunkstartmarker += chunksize

path = "C:\\Users\pault\OneDrive\Documenten\GitHub\input"
path = "/home/afalbrecht/Documents/Leren en Beslissen/"
#path = "S:\OneDrive\Documenten\GitHub\input"

os.chdir(path)

train = pd.read_csv('train.tsv', delimiter='\t', encoding='utf-8')
train = train[train["price"] != 0]  # Drops rows with price = 0
train.index = range(len(train))
print("train size:", train.shape)

#train = train.loc[:100000]
#description_dict, categories_dict, brand_dict = edit_data_make_dicts(train)
# print(len(description_dict), len(categories_dict), len(brand_dict))

train = train.loc[:10000]
start = time.time()

edit_data(train)
print(time.time() - start)
description_dict, categories_dict, brand_dict = make_dictionaries(train)
print(time.time() - start)
print(len(description_dict), len(categories_dict), len(brand_dict))
sparse_matrix = make_sparse_matrix(train, description_dict, categories_dict, brand_dict)
prices = get_price_list(train[:9000])
vec_len = len(description_dict) + len(categories_dict) + len(brand_dict) + 6
# print(description_dict)
neuralnet = make_model(vec_len)


print("made matrix", time.time() - start)

batches = iter_minibatches(sparse_matrix[:9000], prices, chunksize=200)

count = 0
for X_chunk, y_chunk in batches:
    print(count)
    count += 1
    if len(X_chunk) != 0:
        for i in range(5):
            neuralnet.train_on_batch(X_chunk, y_chunk)

valmat = sparse_matrix[9000:].todense()
valprices = get_price_list(train[9000:10000])
print(valmat.shape)
print(valprices.shape)

predicted_prices = neuralnet.predict(valmat)
print('Prices predicted', time.time() - start)

print(valprices.shape)
print(predicted_prices.shape)
print("The score is:", calc_score(valprices, predicted_prices))

print(time.time() -start)

# vector = cat_v + brand_v + item_v + cond_v + [row["shipping"]]
# item_desc = Input(shape=[len(item_desc)], name="item_desc")
# brand = Input(shape=[1], name="brand")
# category = Input(shape=[1], name="category")
# category_name = Input(shape=[X_train["category_name"].shape[1]],
#                       name="category_name")
# item_condition = Input(shape=[1], name="item_condition")
# num_vars = Input(shape=[X_train["num_vars"].shape[1]], name="num_vars")