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

print(data.head())
print(data.index)  

# def get_price_estimate(date, data):
#     date = pd.to_datetime(date, format='%m/%d/%y')  # Assurez-vous que le format correspond à vos données
#     print(f"Converted date: {date}")
#     if date in data.index:
#         return data.loc[date, 'Prices']
#     else:
#         # Interpolation avec np.interp
#         return np.interp(date.timestamp(), data.index.astype(int).values, data['Prices'].values)
def get_price_estimate(date, data):
    date = pd.to_datetime(date, format='%m/%d/%y')  
    if date in data.index:
        return data.loc[date, 'Prices']
    timestamps = data.index.astype('int64') // 1e9  # Timestamps en secondes Unix
    interpolated_price = np.interp(date.timestamp(), timestamps, data['Prices'].values)
    print(f"Interpolated price: {interpolated_price}")
    return interpolated_price
# def get_price_estimate(date, data):
 
#   date = pd.to_datetime(date)
#   if date in data.index:
#     return data.loc[date, 'Prices']
#   else:
#     # Interpolation with the function `np.interp`
#     return np.interp(date.timestamp(), data.index.astype(int).values, data['Prices'].values)
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
    #print(f"Estimated price on date: ${estimate_price(date, data, fitted_model):.2f}") 
    return estimate_price(date, data, fitted_model)



#################################################################################################################



    
def price_gas_contract(
    injection_dates, 
    withdrawal_dates,  
    injection_rates, 
    withdrawal_rates, 
    max_storage, 
 storage_costs):
  
    
    total_value = 0
    total_storage_volume = 0
    injection_costs = 0
    withdrawal_costs = 0
    injection_prices= [seasonal_model_estimation(injection_dates[i]) for i in range(len(injection_dates))]
    print (injection_prices)
    withdrawal_prices= [seasonal_model_estimation(withdrawal_dates[i]) for i in range(len(withdrawal_dates))]
    print( withdrawal_prices)

    # Step 1: Injection phase
    for date, price, rate in zip(injection_dates, injection_prices, injection_rates):
        injection_volume = min(rate, max_storage - total_storage_volume) #we add the maximum possible quantities until exhaustion
        total_storage_volume += injection_volume  #storage uptadte
        injection_costs += injection_volume * price #storage cost
        if total_storage_volume >= max_storage:     #max storage vérification
            break

    # Step 2: Withdrawal phase
    for date, price, rate in zip(withdrawal_dates, withdrawal_prices, withdrawal_rates):
        withdrawal_volume = min(rate, total_storage_volume) 
        total_storage_volume -= withdrawal_volume    
        withdrawal_costs += withdrawal_volume * price  

    # Step 3: Calculate storage costs
    num_months = max(len(injection_dates), len(withdrawal_dates))
    total_storage_costs = num_months * storage_costs

    # Step 4: Compute contract value
    total_value = withdrawal_costs - injection_costs - total_storage_costs

    return total_value


#exemple
injection_dates = ['06/23/2024', '06/24/2024/']
withdrawal_dates = ['12/22/2024/', '12/22/2024']
injection_rates = [500000, 500000]  # MMBtu/day
withdrawal_rates = [500000, 500000]  # MMBtu/day
max_storage = 1e6  # MMBtu
storage_costs = 100000  # $/month

# Call function
contract_value = price_gas_contract(
    injection_dates, withdrawal_dates,
    injection_rates, withdrawal_rates,
    max_storage, storage_costs
)

print(f"The contract value is: ${contract_value:,.2f}")

