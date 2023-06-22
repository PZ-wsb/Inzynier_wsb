import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Wczytaj dane z pliku CSV
import pandas as pd

# Wczytaj dane z pliku CSV
data = pd.read_csv('C:/Users/Zieba/OneDrive/Pulpit/inzynier_wsb/Inzynier_wsb/test.csv', delimiter=';')
print(data.columns)

# Ustaw kolumnę zawierającą daty jako indeks
data['Data'] = pd.to_datetime(data['Data'])
data.set_index('Data', inplace=True)

# Przygotuj dane jako szereg czasowy
ts = data['WIG']

# Zbuduj model ARIMA
model = ARIMA(ts, order=(1, 0, 0))  # Przykładowy order - (p, d, q)

# Dopasuj model do danych
model_fit = model.fit()

# Wykonaj prognozę na przyszłość
forecast = model_fit.forecast(steps=100)  # Przykładowa liczba prognozowanych kroków

print(forecast)
