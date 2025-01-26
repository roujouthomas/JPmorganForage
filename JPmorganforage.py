#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 19:53:03 2024

@author: thomasroujou
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from statsmodels.tsa.holtwinters import ExponentialSmoothing   


data = pd.read_csv('/Users/thomasroujou/Desktop/Nat_Gas.csv', 
                   parse_dates=['Dates'], 
                   index_col='Dates', 
                   date_parser=lambda x: pd.to_datetime(x, format='%m/%d/%y'))


print(data.columns)

def get_price_estimate(date, data):
 
  date = pd.to_datetime(date)
  if date in data.index:
    return data.loc[date, 'Prices']
  else:
    # Interpolation with the function `np.interp`
    return np.interp(date.timestamp(), data.index.astype(int).values, data['Prices'].values)
#P(t)= P1 +(t-t1)/(t2-t1)*(P2-P1) for t<tmax data


test_date = '8/31/22'
print(f"Prix estimé au {test_date} : ${get_price_estimate(test_date, data):.2f}")

# Fit the model( holt-winters exponential Smoothing, additive formule because the variations are constant in magnitude)
model = ExponentialSmoothing(data['Prices'], trend='add', seasonal='add', seasonal_periods=12)
fitted_model = model.fit()

# Forecast for next 12 months
future_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=12, freq='M')
forecast = fitted_model.forecast(steps=12)

# Combine historical and forecast data
forecast_data = pd.DataFrame({'Prices': forecast}, index=future_dates)
complete_data = pd.concat([data, forecast_data])

# Plot historical and forecast prices
plt.figure(figsize=(12, 6))
plt.plot(complete_data.index, complete_data['Prices'], label='Historical + Forecasted Prices', color='blue')
plt.axvline(data.index[-1], color='red', linestyle='--', label='Forecast Start')
plt.title('Natural Gas Prices (Historical and Forecasted)')
plt.xlabel('Dates')
plt.ylabel('Prices')
plt.legend()
plt.grid()
plt.show()
def estimate_price(date, data, model):
    date = pd.to_datetime(date)
    
    if date <= data.index[-1]:
        # Interpolation for historical data
        return get_price_estimate(date, data)
    else:
        # Extrapolation for future data with Holt-Winters
        months_ahead = (date.year - data.index[-1].year) * 12 + (date.month - data.index[-1].month)
        if months_ahead <= 12:
            return model.forecast(steps=12)[months_ahead - 1]
        else:
            raise ValueError("Forecast only available for up to 12 months beyond the last data point.")



def seasonal_model_estimation(date):
    print(f"Estimated price on date: ${estimate_price(date, data, fitted_model):.2f}") 

