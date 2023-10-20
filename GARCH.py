import pandas as pd
import numpy as np
from arch import arch_model

# Wczytanie danych z pliku CSV
data = pd.read_csv('C:/Users/Zieba/OneDrive/Pulpit/inzynier_wsb/Inzynier_wsb/test1.csv', delimiter=';')
print(data.columns)

# Przygotowanie danych do modelu GARCH
returns = data['WIG']
log_returns = np.log(1 + returns)  # Zakładając, że mamy zwroty jako procentowe zmiany

# Tworzenie modelu GARCH(1, 1)
model = arch_model(log_returns, vol='Garch', p=1, q=1)

# Dopasowanie modelu do danych
results = model.fit(disp='off')

# Prognoza na kolejne 5 dni
forecast_horizon = 5
forecasts = results.forecast(start=len(log_returns), horizon=forecast_horizon, reindex=True)  # Ustawienie reindex na True

# Wyświetlenie prognoz na kolejne dni
print(forecasts.mean[-1:])