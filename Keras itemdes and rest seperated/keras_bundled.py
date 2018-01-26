from kerasnoitemdes import *
from kerasonlyitemdes import *

only_itemdes_score, only_itemdes_prices = only_itemdes(10000, 10)
no_itemdes_score, no_itemdes_prices = no_itemdes(10000, 10)

for i in range(len(only_itemdes_prices)):
    print(only_itemdes_prices[i], no_itemdes_prices[i])
print(only_itemdes_score, no_itemdes_score)