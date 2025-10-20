import pandas as pd
import numpy as np
import xgboost as xgb
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold



train_data = pd.read_csv("train.csv")
train_data = train_data.drop('Id', axis=1)


# print(data['SalePrice'].describe())
# plt.figure(figsize=(9, 8))
# sns.distplot(data['SalePrice'], color='g', bins=100, hist_kws={'alpha': 0.4});

X = train_data.drop(columns=['SalePrice'])
y = train_data['SalePrice']

for col in X.select_dtypes(include='object').columns:
    X[col] = X[col].fillna('None').astype('category')

data_dmatrix = xgb.DMatrix(data=X, label=y, enable_categorical=True)





xgb_reg = XGBRegressor(
    objective='reg:squarederror',
    eval_metric='rmse',
    enable_categorical=True
)
parameter_grid = {
    'n_estimators': [100, 400, 800],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.05, 0.1, 0.20],
    'min_child_weight': [1, 10, 100]
    }


cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)

random_search = RandomizedSearchCV(
    estimator=xgb_reg,
    param_distributions=parameter_grid,
    n_iter=10,
    scoring='neg_root_mean_squared_error',
    cv=cv_strategy,
    verbose=2,
    n_jobs=-1
)

random_search.fit(X, y)
model = random_search.best_estimator_

test_data = pd.read_csv("test.csv")
test_data = test_data.drop('Id', axis=1)
X_test = test_data.copy()


for col in X.select_dtypes(include='category').columns:
    if col in X_test.columns:
        X_test[col] = X_test[col].fillna('None').astype('category')

for col in X.select_dtypes(include=['int64','float64']).columns:
    if col in X_test.columns:
        X_test[col] = X_test[col].fillna(X[col].median())

#
y_pred = model.predict(X_test)  # use tuned model



# from sklearn.metrics import mean_squared_error

# rmse = np.sqrt(mean_squared_error(np.log1p(y), np.log1p(y_pred)))
# print("Log-RMSE on training data:", rmse)