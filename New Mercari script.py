import os
import nltk
from funcs import *

path = "C:\\Users\pault\OneDrive\Documenten\GitHub\input"
#path = "/home/afalbrecht/Documents/Leren en Beslissen/"
#path = "/Users/falksinke/LocalDocs/mercari_project"
#path = "S:\OneDrive\Documenten\GitHub\input"

os.chdir(path)

train = pd.read_csv('train.tsv', delimiter='\t', encoding='utf-8')
train = train[train["price"] != 0]  # Drops rows with price = 0
train.index = range(len(train))
print("train size:", train.shape)

start = time.time()
edit_data(train)
print("edit data", time.time() - start)

# description_dict, category_dict, brand_dict = make_dictionaries(train)
# #print(category_dict)
# print(len(description_dict), len(category_dict), len(brand_dict))
# print("make dictionaries", time.time() - start)
# sparse_matrix = make_sparse_matrix(train, description_dict, category_dict, brand_dict)
# print("make sparse matrix", time.time() - start)
# part = sparse_matrix[0:10000].todense()
# print("part", time.time()- start)
# #print(part.shape)

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


print("made matrix", time.time() - start)


neuralnet = MLPRegressor(activation='relu', alpha=1e-04, batch_size=50,
                        beta_1=0.9, beta_2=0.999, early_stopping=False,
                        epsilon=1e-08, hidden_layer_sizes=(40,30,15), learning_rate='constant',
                        learning_rate_init=0.001, max_iter=500, momentum=0.9,
                        nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
                        solver='adam', tol=0.0001, validation_fraction=0.1, verbose=False,
                        warm_start=True)

batches = iter_minibatches(sparse_matrix, prices, chunksize=1000)

count = 0
for X_chunk, y_chunk in batches:
   print(count)
   count += 1
   if len(X_chunk) != 0:
        neuralnet._partial_fit(X_chunk, y_chunk)

valmat = sparse_matrix[999999:].todense()
valprices = get_price_list(train)
print(valmat.shape)
print(valprices.shape)

predicted_prices = neuralnet.predict(valmat)
print('Prices predicted', time.time() - start)

print(valprices.shape)
print(predicted_prices.shape)
print("The score is:", calc_score(valprices, predicted_prices))