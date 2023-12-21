# House Prices XGBRegressor & LightGBMRegressor

### DataSet From [Kaggle - House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

## Overview

- Language: Python v3.9.15
- Package: xgboost, lightgbm
- Model: XGBRegressor, LGBMRegressor
- Loss Function: Mean Squared Error
- Fill in a lot of missing values

## Data Distribution

- Histogram of Values for Each Field

![image](https://github.com/yuhexiong/house-prices-XGBRegressor-LightGBMRegressor-python/blob/main/image/column_count.png)

- Scatter Plot of Each Field Against Price

![image](https://github.com/yuhexiong/house-prices-XGBRegressor-LightGBMRegressor-python/blob/main/image/column_vs_prices.png)

## Correlation Matrix

![image](https://github.com/yuhexiong/house-prices-XGBRegressor-LightGBMRegressor-python/blob/main/image/correlation_matrix.png)

## Result

![image](https://github.com/yuhexiong/house-prices-XGBRegressor-LightGBMRegressor-python/blob/main/image/actual_vs_predicted_prices.png)

- Best Model With Params: XGBRegressor(n_estimators = 2000, reg_lambda = 10, max_depth = 3)
- Mean Squared Error: 2801.275107982908