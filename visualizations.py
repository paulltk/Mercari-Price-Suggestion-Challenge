import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import Counter

path = "C:\\Users\Eigenaar\OneDrive\Documenten\GitHub\input"
os.chdir(path)

train = pd.read_csv('train.tsv', delimiter='\t', encoding='utf-8')
print(train.shape)

# prices = train["price"].tolist()
# print(prices)
# plt.hist(prices, range(0, 150, 2), color="#FFB112")
# plt.xlabel("Price")
# plt.ylabel("Occurrences")
# plt.title("Distribution of prices in training dataset")
# plt.show()

words_list = []
train["item_description"].fillna("missing", inplace=True)
description_list = train["item_description"].tolist()
for sen in description_list:
    for word in sen.split():
        words_list.append(word)
print(len(words_list))
count = Counter(words_list)
print(count)
print(len(count))
words_occurrence = [count[key] for key in count]
print(words_occurrence)
plt.hist(words_occurrence, range(0, 30), color="#FFB112")
plt.xlabel("Occurrence of word")
plt.ylabel("Amount of words")
plt.title("Amount of words per amount of times a word occurs")
plt.show()
