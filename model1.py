import os

import pandas as pd
import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer as DV
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error
import seaborn as sns



def cat_to_dummy(df, cat_cols, test=False, vectorizer=None):
    """
    Convert categorical variables to dummies for either train or test dataset
    Return vectorizer if train
    """
    num_cols = list(set(df.columns) - set(cat_cols))

    cat = df[cat_cols].astype(str)
    num = df[num_cols]

    x_num = num.values

    cat.fillna( 'NA', inplace = True )

    x_cat = cat.T.to_dict().values()

    if test:
        vec_x_cat = vectorizer.transform(x_cat)
        x = np.hstack((x_num, vec_x_cat))
        return x

    else:
        vectorizer = DV(sparse=False)
        vec_x_cat = vectorizer.fit_transform(x_cat)
        x = np.hstack((x_num, vec_x_cat))
        return x, vectorizer


def plot_feature_importance(feature_importance, col_names, first_n):
    """
    plot relative feature importances
    """

    top_n = col_names[np.argsort(feature_importance)[-first_n:]]
    feature_import= feature_importance[np.argsort(feature_importance)][-first_n:]
    feat_import = feature_import/max(feature_import)

    fig = plt.figure(figsize=(8, 8))
    x_ind = np.arange(feat_import.shape[0])
    plt.barh(x_ind, feat_import, height=.3, align='center')
    plt.ylim(x_ind.min() + .5, x_ind.max() + .5)
    plt.yticks(x_ind, top_n, fontsize=14)
    plt.show()

def grid_search_helper(model, grid, x, y, scoring='mean_squared_error'):
    """
    grid search with multi-threading and cross validation
    Input: model, grids, train_x, train_y, scoring method
    Output: Best model
    """

    new_model_gridsearch = GridSearchCV(model,
                                 random_forest_grid,
                                 n_jobs=-1,
                                 verbose=False,
                                 cv=5,
                                 scoring='mean_squared_error')
    new_model_gridsearch.fit(x, y)

    print "best parameters:", new_model_gridsearch.best_params_

    best_model = new_model_gridsearch.best_estimator_

    best_model_pred = best_model.predict(x)
    best_model_MSE = mean_squared_error(y, best_model_pred)
    print "Optimized %f" % best_model_MSE
    return best_model

def feature_engineer(df, trip_quantity, trip_department, holidays=None):
    """
    Feature engineering:
    join total quantity per trip and number of departments visited per trip to the main dataset
    Extract dayofweek and hour as features to predict shopping time
    """

    df = df.merge(trip_quantity, on=['trip_id'], how='left').merge(trip_department,  on=['trip_id'], how='left')
    df.rename(columns={'department_name': 'num_depart'}, inplace=True)
    df['shopping_started_at'] = pd.to_datetime(df['shopping_started_at'])
    # day of week
    df['DOW'] = pd.DatetimeIndex(df['shopping_started_at']).dayofweek
    df['hour'] = pd.DatetimeIndex(df['shopping_started_at']).hour

    if holidays:
        df['if_holiday'] = [1 if x.split()[0] in holidays else 0 for x in df['shopping_started_at']]

    return df

def plot_boxplot(x, y, cols):
    """
    plot dependent variable against all categorical independent variables
    """

    for i, col in enumerate(cols):
        plt.subplot(2, 2, i+1)
        label = x[col].unique()
        x_col = np.array(x[col])
        plt.boxplot([y[x_col==l] for l in label], vert=1)
        plt.xlabel(col)
        plt.ylabel('Shopping_time')
    plt.show()

def dummies(df, col):
    """
    Create dummies for categorical variable
    """
    dummy = pd.get_dummies(df[col], prefix=col)
    df.drop(col, axis=1, inplace=True)
    return pd.concat([df, dummy], axis=1)

if __name__ == '__main__':

    # Step 0 read data
    order = pd.read_csv('data/order_items.csv')
    train = pd.read_csv('data/train_trips.csv')
    test = pd.read_csv('data/test_trips.csv')

    # Step 1 EDA

    # number of quantities per trip
    trip_quantity = order.groupby('trip_id')['quantity'].sum().reset_index()
    # number of departments visited per trip
    trip_department = order.groupby(['trip_id', 'department_name'])['item_id'].count().reset_index().groupby(['trip_id'])['department_name'].count().reset_index()

    x_cols = ['shopper_id', 'fulfillment_model', 'store_id', 'quantity', 'num_depart', 'DOW', 'hour']
    y_col = 'shopping_time'
    # separate categorical variables and numerical variables
    cat_cols = ['shopper_id', 'fulfillment_model', 'store_id', 'DOW', 'hour']
    num_cols = list(set(x_cols) - set(cat_cols))

    # simple feature engineer
    # holidays in US
    holidays = ['2015-09-05', '2015-10-10', '2015-11-11']
    train_order = feature_engineer(train, trip_quantity, trip_department, holidays)
    x_train = train_order[x_cols]


    if_holiday = [1 if x.split()[0] in holidays else 0 for x in train['shopping_started_at']]

    y_train = np.array([x.total_seconds() for x in (pd.to_datetime(train_order['shopping_ended_at']) - pd.to_datetime(train_order['shopping_started_at']))])

    total = pd.concat([x_train, pd.DataFrame(y_train)], axis=1)
    shopper_mean_time = total.groupby('shopper_id')[0].median().reset_index()
    shopper_mean_time.rename(columns={0: 'shopper_mean_time'}, inplace=True)
    x_train = x_train.merge(shopper_mean_time, on=['shopper_id'], how='left')

    x_cols_final = ['fulfillment_model', 'store_id', 'quantity', 'num_depart', 'DOW', 'hour', 'shopper_mean_time']
    cat_cols_final = ['fulfillment_model', 'store_id', 'DOW', 'hour']
    x_train = x_train[x_cols_final]

    # plot boxplot for categorical independent variables and dependent variable
    cat_cols_eda = ['fulfillment_model', 'store_id', 'quantity', 'DOW']
    plot_boxplot(x_train, y_train, cat_cols_eda)

    # plot scatterplot for numerical independent variables and dependent variable
    for i, col in enumerate(num_cols):
        plt.subplot(1, 2, i+1)
        sns.regplot(x_train[col], y_train)
        plt.ylabel('shopping_time')
    plt.show()

    # check colinearity between independent variables
    # pretty good except quantities and number of departments
    sns.heatmap(x_train.corr())
    plt.show()

    # quickly plot feature importances without much tuning
    # number of departments are not suprisingly an important factor as well as dayofweek
    eda_train = x_train.copy()
    for col in cat_cols_final:
        eda_train = dummies(eda_train, col)
    model = RandomForestRegressor()
    model.fit(eda_train, y_train)
    feature_importance =  model.feature_importances_
    col_names = np.array(eda_train.columns)
    plot_feature_importance(feature_importance, col_names, 10)

    # Model fitting

    # finalize features to train
    x_train_final, vector = cat_to_dummy(x_train, cat_cols_final)


    test_order = feature_engineer(test, trip_quantity, trip_department, holidays)
    x_test = test_order[x_cols]
    x_test = x_test.merge(shopper_mean_time, on='shopper_id', how='left')
    x_test = x_test[x_cols_final]
    x_test = x_test.fillna(x_test.mean())

    # finalize features to test using vectors from training
    x_test_final = cat_to_dummy(x_test, cat_cols_final, test=True, vectorizer=vector)

    # grid search with cross validation to find best model
    model_final = RandomForestRegressor()

    #Grid search

    # random forest
    random_forest_grid = {'max_depth': [None, 5],
                      'max_features': ['sqrt'],
                      'min_samples_leaf': [5, 2],
                      'n_estimators': [50, 20],
                      'random_state': [1]}

    best_model = grid_search_helper(model_final, random_forest_grid, x_train_final, y_train)
    # best MSE**0.5/60 = 16.3 mins
    # not very good result
    # compare to the mean shopping time: 41 mins

    # train model after little grid search (due to time limit)
    best_model.fit(x_train_final, y_train)
    # predict y for test cases
    y_pred = best_model.predict(x_test_final)
    # write y_pred to file
    test['shopping_time'] = y_pred
    result = test[['trip_id', 'shopping_time']].sort('trip_id')
    result.to_csv('prediction.csv', index=False)
