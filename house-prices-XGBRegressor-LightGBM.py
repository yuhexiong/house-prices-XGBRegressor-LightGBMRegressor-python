import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

print("train.head()")
print(train.head())
print("train.shape", train.shape)
print("test.shape", test.shape)

# missing value
print("training data missing value")
print(train.isnull().sum().sort_values(ascending=False)[:20])

print('PoolQC')
print(set(train['PoolQC']))
# PoolQC = nan means No Pool
train['PoolQC'] = train['PoolQC'].fillna('No Pool')
print(set(train['PoolQC']))

print()
print('='*50)

print('MiscFeature')
print(set(train['MiscFeature']))
# MiscFeature = nan means No MiscFeature
train['MiscFeature'] = train['MiscFeature'].fillna('No MiscFeature')
print(set(train['MiscFeature']))

print()
print('='*50)

print('Alley')
print(set(train['Alley']))
# Alley = nan means No Alley
train['Alley'] = train['Alley'].fillna('No Alley')
print(set(train['Alley']))

print()
print('='*50)

print('Fence')
print(set(train['Fence']))
# Fence = nan means No Fence
train['Fence'] = train['Fence'].fillna('No Fence')
print(set(train['Fence']))

print()
print('='*50)

print('FireplaceQu')
print(set(train['FireplaceQu']))
# FireplaceQu = nan means No Fireplace
train['FireplaceQu'] = train['FireplaceQu'].fillna('No Fireplace')
print(set(train['FireplaceQu']))

LotFrontage_mean = train['LotFrontage'].mean()
train['LotFrontage'] = train['LotFrontage'].fillna(LotFrontage_mean)

# GarageYrBlt = nan means No Garage
train['GarageYrBlt'] = train['GarageYrBlt'].fillna(0)

print('GarageCond')
print(set(train['GarageCond']))
# GarageCond = nan means No Garage
train['GarageCond'] = train['GarageCond'].fillna('No Garage')
print(set(train['GarageCond']))

print()
print('='*50)

print('GarageFinish')
print(set(train['GarageFinish']))
# GarageFinish = nan means No Garage
train['GarageFinish'] = train['GarageFinish'].fillna('No Garage')
print(set(train['GarageFinish']))

print()
print('='*50)

print('GarageQual')
print(set(train['GarageQual']))
# GarageQual = nan means No Garage
train['GarageQual'] = train['GarageQual'].fillna('No Garage')
print(set(train['GarageQual']))

print()
print('='*50)

print('GarageType')
print(set(train['GarageType']))
# GarageType = nan means No Garage
train['GarageType'] = train['GarageType'].fillna('No Garage')
print(set(train['GarageType']))

print('BsmtExposure')
print(set(train['BsmtExposure']))
# BsmtExposure = nan means No Basement
train['BsmtExposure'] = train['BsmtExposure'].fillna('No Basement')
print(set(train['BsmtExposure']))

print()
print('='*50)

print('BsmtFinType2')
print(set(train['BsmtFinType2']))
# BsmtFinType2 = nan means No Basement
train['BsmtFinType2'] = train['BsmtFinType2'].fillna('No Basement')
print(set(train['BsmtFinType2']))

print()
print('='*50)

print('BsmtFinType1')
print()
print(set(train['BsmtFinType1']))
# BsmtFinType1 = nan means No Basement
train['BsmtFinType1'] = train['BsmtFinType1'].fillna('No Basement')
print(set(train['BsmtFinType1']))

print()
print('='*50)

print('BsmtQual')
print(set(train['BsmtQual']))
# BsmtQual = nan means No Basement
train['BsmtQual'] = train['BsmtQual'].fillna('No Basement')
print(set(train['BsmtQual']))

print()
print('='*50)

print('BsmtCond')
print(set(train['BsmtCond']))
# BsmtCond = nan means No Basement
train['BsmtCond'] = train['BsmtCond'].fillna('No Basement')
print(set(train['BsmtCond']))

print('MasVnrType')
print(set(train['MasVnrType']))
# MasVnrType = nan means No Basement
train['MasVnrType'] = train['MasVnrType'].fillna('None')
print(set(train['MasVnrType']))

train['MasVnrArea'] = train['MasVnrArea'].fillna(0)
train['Electrical'] = train['Electrical'].fillna(train['Electrical'].mode()[0])

print("sum(train.isnull().sum())", sum(train.isnull().sum()))

print("testing data missing value")
print(test.isnull().sum().sort_values(ascending=False)[:35])

test['PoolQC'] = test['PoolQC'].fillna('No Pool')
test['MiscFeature'] = test['MiscFeature'].fillna('No MiscFeature')
test['Alley'] = test['Alley'].fillna('No Alley')
test['Fence'] = test['Fence'].fillna('No Fence')
test['FireplaceQu'] = test['FireplaceQu'].fillna('No Fireplace')
test['LotFrontage'] = test['LotFrontage'].fillna(LotFrontage_mean)

test['GarageYrBlt'] = test['GarageYrBlt'].fillna(0)
test['GarageQual'] = test['GarageQual'].fillna('No Garage')
test['GarageFinish'] = test['GarageFinish'].fillna('No Garage')
test['GarageCond'] = test['GarageCond'].fillna('No Garage')
test['GarageType'] = test['GarageType'].fillna('No Garage')

test['BsmtCond'] = test['BsmtCond'].fillna('No Basement')
test['BsmtQual'] = test['BsmtQual'].fillna('No Basement')
test['BsmtExposure'] = test['BsmtExposure'].fillna('No Basement')
test['BsmtFinType2'] = test['BsmtFinType2'].fillna('No Basement')
test['BsmtFinType1'] = test['BsmtFinType1'].fillna('No Basement')

test['MasVnrType'] = test['MasVnrType'].fillna('None')
test['MasVnrArea'] = test['MasVnrArea'].fillna(0)

# columns which training data no na
test['MSZoning'] = test['MSZoning'].fillna(train['MSZoning'].mode()[0])
test['BsmtFullBath'] = test['BsmtFullBath'].fillna(0)
test['BsmtHalfBath'] = test['BsmtHalfBath'].fillna(0)
test['Functional'] = test['Functional'].fillna('Typ')
test['Utilities'] = test['Utilities'].fillna(train['Utilities'].mode()[0])
test['Exterior1st'] = test['Exterior1st'].fillna(train['Exterior1st'].mode()[0])
test['Exterior2nd'] = test['Exterior2nd'].fillna(train['Exterior2nd'].mode()[0])
test['SaleType'] = test['SaleType'].fillna(train['SaleType'].mode()[0])
test['GarageCars'] = test['GarageCars'].fillna(0)
test['GarageArea'] = test['GarageArea'].fillna(0)
test['TotalBsmtSF'] = test['TotalBsmtSF'].fillna(0)
test['BsmtUnfSF'] = test['BsmtUnfSF'].fillna(0)
test['BsmtFinSF2'] = test['BsmtFinSF2'].fillna(0)
test['BsmtFinSF1'] = test['BsmtFinSF1'].fillna(0)
test['KitchenQual'] = test['KitchenQual'].fillna(train['KitchenQual'].mode()[0])

print("sum(test.isnull().sum())", sum(test.isnull().sum()))

train2 = train.copy()
for col in train2.columns:
    le = LabelEncoder()
    le.fit(train2[str(col)])
    train2[str(col)] = le.transform(train2[str(col)])

plt.rc('font', size=8)
fig = plt.figure(figsize = (20,20))
ax = fig.gca()
train2.hist(ax=ax, color='sandybrown')
plt.show()

plt.rc('font', size=5)
fig, ax = plt.subplots(nrows=10, ncols=8, figsize=(20, 40))
n = 0
for row in range(10):
    for col in range(8):
        colName = train.columns[n]
        ax[row][col].scatter(train[colName], train['SalePrice'], color='sandybrown', s=5)
        n += 1
plt.show()

# correlation matrix
correlation_matrix = train.corr()
plt.subplots(figsize=(12,9))
# cmap = sns.color_palette("dark:salmon_r", as_cmap=True)
cmap = sns.color_palette("YlOrBr", as_cmap=True)
sns.heatmap(correlation_matrix, vmax=0.9, square=True, cmap=cmap)
plt.show()

# label encoding
test_id = test['Id']
combine = pd.concat([train,test],axis=0)
print(combine.head())
print("combine.shape", combine.shape)

for col in combine.columns:
    le = LabelEncoder()
    le.fit(combine[str(col)])
    combine[str(col)] = le.transform(combine[str(col)])

train_label = combine[:len(train)]
test_label = combine[len(train):]
print(train_label.shape)
print(test_label.shape)

print("train_label.head()")
print(train_label.head())

y_train = train_label['SalePrice']
train_label = train_label.drop(columns = ['Id', 'SalePrice'])
test_label = test_label.drop(columns = ['Id', 'SalePrice'])

X_train, X_valid, y_train, y_valid = train_test_split(train_label, 
                            y_train, test_size=0.2, random_state=0)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_valid = scaler.transform(X_valid)

print(X_train.shape)
print(X_valid.shape)

# XGB Regressor
nSet = [2000, 4000, 6000]
lamb = [10, 30, 50]
depths = [3, 5, 10]
for n in nSet:
    for l in lamb:
        for d in depths:

            xgbRegressor = xgb.XGBRegressor(n_estimators = n, reg_lambda = l, max_depth = d)
            xgbRegressor.fit(X_train, y_train)

            y_valid_pred = xgbRegressor.predict(X_valid)

            print('n_estimators =', n, 'reg_lambda', l,'max_depth =', d)
            print(f"Mean Squared Error: {mean_squared_error(y_valid, y_valid_pred)}")
            print(f"R2: {round(r2_score(y_valid, y_valid_pred), 2)}")
            print('='*50)

# Light GBM
leaves = [1000, 3000, 5000]
lamb = [10, 30, 50]
for leaf in leaves:
    for l in lamb:

        lgbmRegressor = lgb.LGBMRegressor(boosting_type='gbdt',
                                objective='regression', 
                                num_leaves=leaf,
                                learning_rate=0.05,
                                max_depth=-1,
                                reg_lambda=l)
        lgbmRegressor.fit(X_train, y_train)

        y_valid_pred = lgbmRegressor.predict(X_valid)

        print('n_estimators =', n, 'reg_lambda', l,'max_depth =', d)
        print(f"Mean Squared Error: {mean_squared_error(y_valid, y_valid_pred)}")
        print(f"R2: {round(r2_score(y_valid, y_valid_pred), 2)}")
        print('='*50)

# choose XGBRegressor
xgbRegressor = xgb.XGBRegressor(n_estimators = 2000, reg_lambda = 10, max_depth = 3)
xgbRegressor.fit(X_train, y_train)
y_valid_pred = xgbRegressor.predict(X_valid)

mse = mean_squared_error(y_valid, y_valid_pred)
print(f'Mean Squared Error: {mse}')

plt.figure(figsize=(8, 6))
plt.scatter(y_valid, y_valid_pred, color='sandybrown', alpha=0.7)
plt.title('Actual vs Predicted Prices')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.show()

y_test_pred = xgbRegressor.predict(test_label)
submission = pd.DataFrame({
    'Id': test_id,
    'SalePrice': y_test_pred
})

submission.to_csv('data/submission.csv',index=False)