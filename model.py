from math import log, exp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor as GB
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.metrics import mean_squared_error as MSE
from sklearn.cross_validation import cross_val_score

def dummies(df_original, cols):
    """
    Create dummies for categorical variable
    """
    df = df_original.copy()
    for col in cols:
        dummy = pd.get_dummies(df[col], prefix=col)
        df.drop(col, axis=1, inplace=True)
        df = pd.concat([df, dummy], axis=1)
    return df

def feature_importance(df_x, y):
    model = RF()
    model.fit(df_x, y)
    feature_importance =  model.feature_importances_
    plot_feature_importance(feature_importance, np.array(df_x.columns), 10)

def plot_feature_importance(feature_importance, col_names, first_n):
    """
    plot relative feature importances
    """

    print "Plotting 10 most important features given by random forest..."
    top_n = col_names[np.argsort(feature_importance)[-first_n:]]
    feature_import= feature_importance[np.argsort(feature_importance)][-first_n:]
    feat_import = feature_import/max(feature_import)

    fig = plt.figure(figsize=(8, 8))
    x_ind = np.arange(feat_import.shape[0])
    plt.barh(x_ind, feat_import, height=.3, align='center')
    plt.ylim(x_ind.min() + .5, x_ind.max() + .5)
    plt.yticks(x_ind, top_n, fontsize=14)
    plt.title("Feature importances for 10 most important features")
    plt.show()

def cat_to_dummy(df, cat_cols, test=False, vectorizer=None):
    """
    Convert categorical variables to dummies for either train or test dataset
    Return vectorizer if train
    """
    num_cols = list(set(df.columns) - set(cat_cols))
    cat = df[cat_cols].astype(str)
    num = df[num_cols]
    x_num = num.values
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


if __name__ == '__main__':

    train_x = pd.read_csv("indeed_data_science_exercise/train_features_2013-03-07.csv")
    test_x = pd.read_csv("indeed_data_science_exercise/test_features_2013-03-07.csv")
    train_y = pd.read_csv("indeed_data_science_exercise/train_salaries_2013-03-07.csv")

    train = pd.merge(train_x, train_y, how='inner', on='jobId')

    # name categorical and numerical variables and target
    cat_cols = ['companyId', 'jobType', 'degree', 'major', 'industry']
    num_cols = ['yearsExperience', 'milesFromMetropolis']
    y_col = 'salary'

    # check number of unique values for each categorical variable
    for col in cat_cols:
        print col, len(train_x[col].unique())
        print train_x[col].unique()

    # no missing values -- OK
    train_x.count()
    # check outliers -- OK
    train_x.describe()

    # check relationship between categorical variables and target -- OK
    for x_col in cat_cols:
        fig, ax = plt.subplots()
        x_labels = train[x_col].unique()
        plt.boxplot([train[y_col][train[x_col]==l] for l in x_labels], vert=1)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        xtickNames = ax.set_xticklabels(x_labels)
        plt.setp(xtickNames)
        plt.show()

    # check relationship between numerical variables and target
    # Found some salary==0, removed -- OK
    fig, axes = plt.subplots(nrows=1, ncols=2)
    train.plot(ax=axes[0], kind='scatter', x=num_cols[0], y=y_col)
    train.plot(ax=axes[1], kind='scatter', x=num_cols[1], y=y_col)
    plt.show()

    # Removed rows with 0 salary
    train_clean = train[train[y_col]>0]
    train_clean.pop('jobId')
    y = train_clean.pop(y_col)

    # Naive random forest and feature importances
    train_dummy = dummies(train_clean, cat_cols)
    train_dummy.drop('jobId', axis=1, inplace=True)
    feature_importance(train_dummy, y)

    # check correlation between numerical features -- OK, no natrual causation
    sns.heatmap(train.corr())
    plt.show()

    train_fit, vector = cat_to_dummy(train_clean, cat_cols)
    models = [RF(), LR(), GB()]
    # X_train, X_test, y_train, y_test = train_test_split(train_fit, y, test_size=0.2, random_state=1)
    # MSEs = []
    # for model in models:
    #     model.fit(X_train, y_train)
    #     y_pred = model.predict(X_test)
    #     MSEs.append(MSE(y_test, y_pred))
    #
    # for model, mse in zip(models, MSEs):
    #     print model.__class__.__name__, mse
    for model in models:
        print model.__class__.__name__
        print cross_val_score(model, train_fit, y, cv=3, scoring='mean_squared_error')

    # Tried taking log of the salary, but the MSE is even higher.
    # log_y = [log(target) for target in y]
    # log_X_train, log_X_test, log_y_train, log_y_test = train_test_split(train_fit, log_y, test_size=0.2, random_state=1)
    # model = GB()
    # model.fit(log_X_train, log_y_train)
    # log_y_pred = model.predict(log_X_test)
    # y_pred = [exp(target) for target in log_y_pred]
    # y_test = [exp(target) for target in log_y_test]
    # MSE(y_test, y_pred)

    # from the naive model selection, Gradient Boosting outperforms other models.
    # If given longer time, gridsearchCV can be used to tune model better.
    final_model = GB()
    final_model.fit(train_fit, y)

    # Testing
    test_jobId = test_x.pop('jobId')
    test_x_final = cat_to_dummy(test_x, cat_cols, test=True, vectorizer=vector)
    final_pred = final_model.predict(test_x_final)
    d = {'salary': final_pred}
    result = pd.DataFrame(d, index=test_jobId)
    result.to_csv("indeed_data_science_exercise/test_salaries_2013-03-07.csv")
