import pandas as pd
from nltk.stem import WordNetLemmatizer
import re
import string
from sklearn.decomposition import PCA


def fit_PCA(matrix, ratio):
    """ Fit PCA to matrix to a ratio between 0 and 1 and return pca with SVD in it"""
    pca = PCA(n_components=int(matrix.shape[1]*ratio))
    pca_mat = pca.fit_transform(matrix)
    return pca_mat, pca

def test_PCA(matrix, pca):
    """ Run PCA on a matrix using a previously calculated SVD"""
    return pca.transform(matrix)

def edit_data(data):
    split_categories(data)
    fill_missing_items(data)
    change_item_description(data)

# every item (except a few NaN's) have three categories. Here we make a list with those three categories.
def split_categories(data):
    data["cat_list"] = ""
    for index, row in data.iterrows():
        if row["category_name"] == row["category_name"]:
            catlist = [row["category_name"].split("/")[0],
                       row["category_name"].split("/")[1],
                       row["category_name"].split("/")[2]]
            # print(index, catlist)
            data.set_value(index, "cat_list", catlist)
        # print(row)
    return data

#replaces NaN, which is not a string, with "missing"
def fill_missing_items(data):
    data["brand_name"].fillna("missing", inplace=True)
    data["item_description"].fillna("missing", inplace=True)

# replaces the item description with
def change_item_description(data):
    wnlm = WordNetLemmatizer()
    stopwords = set("""rm i me my myself we our ours ourselves you your yours yourself yourselves he him 
            his himself she her hers herself it its itself they them their theirs themselves what which who whom 
            this that these those am is are was were be been being have has had having do does did doing a an the 
            and but if or because as until while of at by for with about against between into through during before 
            after above below to from up down in out on off over under again further then once here there when where 
            why how all any both each few more most other some such no nor not only own same so than too very s t can 
            will just don should now""".split(" "))
    for index, row in data.iterrows():
        des = row["item_description"]
        des = des.lower()
        txt = re.sub("[^0-9a-zA-Z ]", "", des)
        words = [wnlm.lemmatize(w) for w in txt.split(" ") if w not in stopwords and len(w) > 2]
        data.set_value(index, "item_description", words)
