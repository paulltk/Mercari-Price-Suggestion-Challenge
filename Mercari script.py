from mercarifunc import *

path = "C:\\Users\pault\OneDrive\Documenten\GitHub\input" # pauls path
#path = "/home/afalbrecht/Documents/Leren en Beslissen/" #add your path here, dit is een test
os.chdir(path)

train = pd.read_csv('train.tsv', delimiter='\t', encoding='utf-8')
print("train size:", train.shape)

test = pd.read_csv('test.tsv', delimiter='\t', encoding='utf-8')
print("test size:", test.shape)

# sample = train.loc[0:10000]
# change_item_description(sample)
# split_categories(sample)
# fill_missing_brand_names(sample)
# matrix = instances_into_vectors(sample)
# print(matrix)

data = train.loc[0:1000]

item_des, categories, brand_names = lists_for_vector(data)
item_dict, cat_dict, brand_dict = list_to_dict(item_des), list_to_dict(categories), list_to_dict(brand_names)
