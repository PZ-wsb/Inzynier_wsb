import pandas as pd

data = pd.read_csv('C:/Users/Zieba/OneDrive/Pulpit/inzynier_wsb/Inzynier_wsb/test.csv', delimiter=';')
#data['Data'] = pd.to_datetime(data['Data'])
#data.set_index('Data', inplace=True)

print(data)
