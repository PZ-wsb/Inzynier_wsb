import pandas as pd
import numpy as np
from arch import arch_model

# Wczytanie danych z pliku CSV
data = pd.read_csv('C:/Users/Zieba/OneDrive/Pulpit/inzynier_wsb/Inzynier_wsb/test1.csv', delimiter=';')

# Przygotowanie danych do modelu GARCH
returns = data['WIG']
log_returns = np.log(1 + returns)

# Przeskalowanie danych y
log_returns_scaled = log_returns * 10

# Tworzenie modelu GARCH(1, 1)
model = arch_model(log_returns_scaled, vol='Garch', p=1, q=1)

# Dopasowanie modelu do danych
results = model.fit(disp='off')

# Prognoza na jeden dzień do przodu
forecast_horizon = 1
last_date = data.index[-1]  # Pobranie ostatniej daty z danych

# Utworzenie próbki dla prognozy
forecast_sample = log_returns_scaled[-model.volatility.start:]
forecast_mean = results.forecast(start=forecast_sample.index[0], horizon=forecast_horizon).mean

# Przywrócenie oryginalnej skali prognozowanego zwrotu
forecasted_return = forecast_mean.iloc[-1] / 10

# Obliczenie przyszłej wartości indeksu
last_index_value = data['WIG'].iloc[-1]
forecasted_index_value = last_index_value * np.exp(forecasted_return)

# Wyświetlenie prognozy
print("Last Date:", last_date)
print("Forecasted Return for Next Day:", forecasted_return)
print("Forecasted Index Value for Next Day:", forecasted_index_value)
