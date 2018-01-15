# foo = "missing cats"
#
# for x in foo:
#     print(x)
#
# with open('test.csv', 'w') as file:
#     file.write('test')
path = "C:\\Users\pault\OneDrive\Documenten\GitHub\input" # pauls path
#path = "/home/afalbrecht/Documents/Leren en Beslissen/" #add your path here, dit is een test
os.chdir(path)

train = pd.read_csv('train.tsv', delimiter='\t', encoding='utf-8')
data = train.loc[:11000]
prices = get_price_list(data[:10000])