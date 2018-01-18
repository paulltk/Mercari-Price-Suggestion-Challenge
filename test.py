# # # import numpy as np
# # # import pandas as pd
# # # import os
# # # import statistics
# # # import heapq
# # # import re, string
# # # import time
# # # import math
# # # from collections import Counter
# # # from sklearn.decomposition import PCA
# # # from nltk.stem import WordNetLemmatizer
# # # from sklearn.neural_network import MLPClassifier
# # # from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
# # # from sklearn.cross_validation import train_test_split
# # # import matplotlib.pyplot as plt
# # # from preprocessing import *
# # # #from vectorizenoitemdes import *
# # # from vectorize import *
# # # from keras.preprocessing.text import Tokenizer
# # #
# # # def split_categories(data):
# # #     data["cat_list"] = ""
# # #     for index, row in data.iterrows():
# # #         if row["category_name"] == row["category_name"]:
# # #             catlist = [row["category_name"].split("/")[0],
# # #                        row["category_name"].split("/")[1],
# # #                        row["category_name"].split("/")[2]]
# # #             # print(index, catlist)
# # #             data.set_value(index, "cat_list", catlist)
# # #         # print(row)
# # #     return data
# # #
# # # def fill_missing_items(data):
# # #     data["brand_name"].fillna("missing", inplace=True)
# # #     data["item_description"].fillna("missing", inplace=True)
# # #
# # # def change_item_description(data):
# # #     wnlm = WordNetLemmatizer()
# # #     stopwords = set("""rm i me my myself we our ours ourselves you your yours yourself yourselves he him
# # #             his himself she her hers herself it its itself they them their theirs themselves what which who whom
# # #             this that these those am is are was were be been being have has had having do does did doing a an the
# # #             and but if or because as until while of at by for with about against between into through during before
# # #             after above below to from up down in out on off over under again further then once here there when where
# # #             why how all any both each few more most other some such no nor not only own same so than too very s t can
# # #             will just don should now""".split(" "))
# # #     train.item_description.str.lower()
# # #     for index, row in data.iterrows():
# # #         des = row["item_description"]
# # #         txt = re.sub("[^0-9a-zA-Z ]", "", des)
# # #         words = [wnlm.lemmatize(w) for w in txt.split(" ") if w not in stopwords and len(w) > 2]
# # #         data.set_value(index, "item_description", words)
# # #
# # # def item_des_to_token(data):
# # #        raw_text = np.hstack([train.item_description])
# # #        tok_raw = Tokenizer()
# # #        tok_raw.fit_on_texts(raw_text)
# # #        train["seq_item_description"] = tok_raw.texts_to_sequences(train.item_description)
# # #
# # #
# # # path = "C:\\Users\pault\OneDrive\Documenten\GitHub\input"
# # # os.chdir(path)
# # #
# # # train = pd.read_csv('train.tsv', delimiter='\t', encoding='utf-8')
# # # print("train size:", train.shape)
# # # test = pd.read_csv('test.tsv', delimiter='\t', encoding='utf-8')
# # # print("test size:", test.shape)
# # #
# # # data = train.loc[:100000]
# # #
# # # start = time.time()
# # # print(start)
# # # split_categories(data)
# # # print("split cat", time.time() - start)
# # # fill_missing_items(data)
# # # print("fill missing items", time.time() - start)
# # # change_item_description(data)
# # # print("change item des", time.time() - start)
# # # item_des_to_token(data)
# # # print("item des to token", time.time() - start)
# # #
# # # print(data.head(5))
# #
# #
# # import pandas as pd
# # import os
# # path = "C:\\Users\pault\OneDrive\Documenten\GitHub\input"
# # os.chdir(path)
# # train = pd.read_csv('train.tsv', delimiter='\t', encoding='utf-8')
# # print("train size:", train.shape)
# # data = train.loc[:100]
# #
# #
# # data["item_description"] = data["item_description"].str.split()
# #
# # for i in data["item_description"]:
# #     print(i)
# #
# # print(data.head(5))
# #
# #
# # def words_list(data):
# #     words = []
# #     data["item_description"] = data["item_description"].str.split()
# #     descr_list = data["item_description"].tolist()
# #     for sen in descr_list:
# #         for word in sen:
# #             if len(word) > 3:
# #                 words.append(re.sub("[^0-9a-z]", "", word.lower()))
# #     count = Counter(words)
# #     return set([w for w in words if count[w] > 10])  # list with words
# #
# # def make_dicts(data):
# #     start = time.time()
# #     data["category_name"] = data["category_name"].str.split("/")
# #     print("split cat", time.time() - start)
# #     fill_missing_items(train)
# #     print("fill missing", time.time() - start)
# #     words = words_list(data)
# #     print("make words", time.time() - start)
# #     categories = []
# #     brand_names = []
# #     for cats in data["category_name"]:
# #         if cats == cats:
# #             categories += cats
# #     for brand in data["brand_name"]:
# #         brand_names.append(brand)
# #     print("append", time.time() - start)
# #     return list_to_dict(words), list_to_dict(list(set(categories))), list_to_dict(list(set(brand_names)))
# #
# dict = {"a": 6, "b": 10}
#
# v = [0] * 20
#
# a = ["g", "a"]
#
# for i in a:
#     try: v[dict[i]] = 1
#     except: pass
#
# print(v)
#

m = []

a = [1, 2]
b = [3, 4]

m.append(a)
m.append(b)

print(m)