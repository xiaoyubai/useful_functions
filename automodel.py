import os
import math

import pandas as pd
import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer as DV
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error, recall_score, precision_score, accuracy_score
from sklearn.cross_validation import train_test_split
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.svm import SVC
import sklearn.metrics as skm
from sklearn.preprocessing import OneHotEncoder


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# basic models:
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV
# language processing
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import OneHotEncoder
import cPickle as pickle
# testing
from sklearn.metrics import recall_score
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split

class AutomatedClassifier(object):
    def __init__(self, model=RF()):
        self.model = model

    def check_col_type(self, unique_categories=100):
        possible_cat_cols = []
        possible_num_cols = []
        for col in self.train_df.columns:
            try:
                if len(self.train_df[col].unique()) < unique_categories:
                    possible_cat_cols.append(col)
                elif (self.train_df[col].dtype == np.float64 or self.train_df[col].dtype == np.int64):
                    possible_num_cols.append(col)
            except:
                pass
        if not self.cat_cols:
            self.cat_cols = possible_cat_cols

        if not self.num_cols:
            self.num_cols = possible_num_cols

        if self.y_col in self.cat_cols:
            self.cat_cols.remove(self.y_col)

    def fill_na(self, df):
        for col in self.cat_cols:
            df[col].fillna('NA', inplace=True)
        for col in self.num_cols:
            df[col].fillna(df[col].mean(), inplace=True)
        return df

    def EDA(self):
        self.plot_boxplot()
        self.plot_countplot()
        self.plot_heat_map()
        self.feature_importance()

    def fit(self, df, y_col, cat_cols=None, num_cols=None, oversampling=False, target=None):
        self.train_df = df
        self.y_col = y_col
        self.x_train = None
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.check_col_type()
        self.train_df = self.fill_na(self.train_df)
        self.y_train = df[self.y_col]
        self.cat_to_dummy(df, test=False)
        self.EDA()
        if oversampling:
            self.x_train, self.y_train = smote(self.x_train, self.y_train, target=target, k=None)

        self.model.fit(self.x_train, self.y_train)
        self.cross_val()

    def predict(self, df):
        self.x_test = None
        df = self.fill_na(df)
        self.cat_to_dummy(df, test=True)
        return self.model.predict(self.x_test)

    def score(self, y_true, y_pred):

        self.test_recall = recall_score(y_true, y_pred)
        self.test_precision = precision_score(y_true, y_pred)
        self.test_accuracy = accuracy_score(y_true, y_pred)
        print "recall: %.2f\n precision: %.2f\n accuracy: %.2f" % (self.test_recall, self.test_precision, self.test_accuracy)

    def cat_to_dummy(self, df, test=False):
        """
        Convert categorical variables to dummies for either train or test dataset
        Return vectorizer if train
        """
        cat = df[self.cat_cols].astype(str)
        num = df[self.num_cols]

        x_num = num.values
        x_cat = cat.T.to_dict().values()

        if test:
            vec_x_cat = self.vectorizer.transform(x_cat)
            self.x_test = np.hstack((x_num, vec_x_cat))

        else:
            self.vectorizer = DV(sparse=False)
            vec_x_cat = self.vectorizer.fit_transform(x_cat)
            self.x_train = np.hstack((x_num, vec_x_cat))

    def dummies(self, df, col):
        """
        Create dummies for categorical variable
        """
        dummy = pd.get_dummies(df[col], prefix=col)
        df.drop(col, axis=1, inplace=True)
        return pd.concat([df, dummy], axis=1)

    def feature_importance(self):
        eda_train = self.train_df[self.cat_cols + self.num_cols].copy()
        for col in self.cat_cols:
            eda_train = self.dummies(eda_train, col)

        model = RF()
        model.fit(eda_train, self.y_train)
        feature_importance =  model.feature_importances_
        col_names = np.array(eda_train.columns)
        self.plot_feature_importance(feature_importance, col_names, 10)

    def plot_feature_importance(self, feature_importance, col_names, first_n):
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

    def plot_heat_map(self):
        print "Plotting correlation heatmap between features..."
        labels = self.cat_cols + self.num_cols
        sns.heatmap(self.train_df[labels].corr())
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.show()

    def plot_dimensions(self, cols_to_plot):
        num_plots = len(cols_to_plot)
        # num_row = int(round(np.sqrt(num_plots)))
        # num_col = num_plots / num_row + 1
        num_col = 3
        num_row = num_plots / num_col + 1
        return num_row, num_col

    def plot_boxplot(self):
        """
        plot dependent variable against all numerical independent variables
        """

        print "Plotting possible numerical features with target..."
        num_row, num_col = self.plot_dimensions(self.num_cols)
        y_labels = self.train_df[self.y_col].unique()
        fig = plt.figure()
        for i, x_col in enumerate(self.num_cols):
            ax = fig.add_subplot(num_row, num_col, i+1)
            top = sorted(self.train_df[x_col])[int(0.95*len(self.train_df))]
            bottom = sorted(self.train_df[x_col])[int(0.05*len(self.train_df))]
            plt.boxplot([self.train_df[x_col][self.y_train==l] for l in y_labels], vert=1)
            plt.xlabel(self.y_col)
            plt.ylabel(x_col)
            plt.ylim(bottom, top)
            xtickNames = ax.set_xticklabels(y_labels)
            plt.setp(xtickNames)
        plt.show()

    def plot_countplot(self):
        """
        plot dependent variable against all categorical independent variables
        """

        print "Plotting possible categorical features with target..."
        col_to_plot = []
        for col in self.cat_cols:
            if len(self.train_df[col].unique()) < 20:
                col_to_plot.append(col)
        num_row, num_col = self.plot_dimensions(col_to_plot)
        for i, col in enumerate(col_to_plot):
            plt.subplot(num_row, num_col, i+1)
            sns.countplot(x=col, hue=self.y_col, data=self.train_df)
        plt.show()

    def cross_val(self):
        self.recall = cross_val_score(self.model, self.x_train, self.y_train, scoring='recall').mean()
        self.precision = cross_val_score(self.model, self.x_train, self.y_train, scoring='precision').mean()
        self.accuracy = cross_val_score(self.model, self.x_train, self.y_train, scoring='accuracy').mean()


    def plot_profit_curve(self, costbenefit, y_test, ROC=True):
        models = [RF(), LR(), GBC()]
        print "Plotting Profit Curve for different models..."
        for profit_curve_model in models:
            profit_curve_model.fit(self.x_train, self.y_train)
            y_pred_prob = profit_curve_model.predict_proba(self.x_test)[:, 1]
            profits = profit_curve(costbenefit, y_pred_prob, y_test)
            percentages = np.arange(0, 100, 100. / len(profits))

            plt.plot(percentages, profits, label=profit_curve_model.__class__.__name__)
            plt.title("Profit Curve")
            plt.xlabel("Pipercentage of test instances")
            plt.ylabel("Profit")
            plt.legend(loc='lower right')
        plt.show()

        if ROC:
            y_pred_probas = self.model.predict_proba(self.x_test)[:, 1]
            roc_curve(y_pred_probas, y_test)
            plt.show()


def profit_curve(cb, predict_probas, labels):
    final_profits = []

    sorted_predict_probas = sorted(predict_probas)
    # Sort instances by their prediction strength (the probabilities)

    for prob in sorted_predict_probas:
        labels_predict = np.zeros(len(labels))
        threshold = prob
        labels_predict[predict_probas > threshold] = 1
        standard_matrix = standard_confusion_matrix(labels, labels_predict)
        profit_matrix = standard_matrix * cb
        profit = np.sum(profit_matrix) / len(labels)
        final_profits.append(profit)

    return final_profits

def standard_confusion_matrix(y_true, y_predict):
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    ones = np.ones(len(y_true))
    tp = np.sum(ones[(y_true==1) & (y_predict==1)])
    fp = np.sum(ones[(y_true==0) & (y_predict==1)])
    fn = np.sum(ones[(y_true==1) & (y_predict==0)])
    tn = np.sum(ones[(y_true==0) & (y_predict==0)])
    return np.array([[tp, fp], [fn, tn]])

def smote(X, y, target, k=None):
    """
    INPUT:
    X, y - your data
    target - the percentage of positive class
             observations in the output
    k - k in k nearest neighbors

    OUTPUT:
    X_oversampled, y_oversampled - oversampled data

    `smote` generates new observations from the positive (minority) class:
    For details, see: https://www.jair.org/media/953/live-953-2037-jair.pdf
    """
    y_positive = y[y==1]
    y_negative = y[y==0]
    X_positive = X[y==1]
    X_negative = X[y==0]
    # fit a KNN model
    kNN = NearestNeighbors(n_neighbors=k)
    kNN.fit(X_positive, y_positive)

    # determine how many new positive observations to generate
    n_positive = len(y[y==1])
    n_total = len(y)
    if n_positive / float(n_total) < target:
        y_add = math.ceil((target * n_total - n_positive) / (1 - target))

    # generate synthetic observations
    random_indices = np.random.choice(range(len(X_positive)), y_add, replace=True)
    replicate_points = X_positive[random_indices]
    new_X = []
    for point in replicate_points:
        neighbors = kNN.kneighbors([point], 5, return_distance=False)
        chosen_neighbor = np.random.choice(neighbors[0], 1)
        p = np.random.rand(len(point))
        new_x = point * p + X_positive[chosen_neighbor][0] * (1 - p)
        # print point
        # print X_positive[chosen_neighbor]
        # print new_x
        # print p
        # break
        new_X.append(new_x)

    # combine synthetic observations with original observations
    new_X = np.array(new_X)
    X_smoted = np.concatenate((X, new_X), axis=0)
    new_y = np.ones(y_add)
    y_smoted = np.concatenate((y, new_y), axis=0)
    return X_smoted, y_smoted

def roc_curve(probabilities, labels):
    '''
    INPUT: numpy array, numpy array
    OUTPUT: list, list, list
    Take a numpy array of the predicted probabilities and a numpy array of the
    true labels.
    Return the True Positive Rates, False Positive Rates and Thresholds for the
    ROC curve.
    '''
    # function ROC_curve(probabilities, labels):
    # Sort instances by their prediction strength (the probabilities)
    # For every instance in increasing order of probability:
    #     Set the threshold to be the probability
    #     Set everything above the threshold to the positive class
    #     Calculate the True Positive Rate (aka sensitivity or recall)
    #     Calculate the False Positive Rate (1 - specificity)
    # Return three lists: TPRs, FPRs, thresholds
    print "Plotting ROC Curve..."
    sort_prob = np.sort(probabilities)
    predictions = []
    TPRs = []
    FPRs = []
    ones = np.ones(probabilities.size)
    labels = np.array(labels)
    for prob in sort_prob:
        prediction = (probabilities > prob) * 1
        TP = np.sum(ones[(prediction==1)&(labels==1)])
        FP = np.sum(ones[(prediction==1)&(labels==0)])
        TPR = TP/(np.sum(labels))
        FPR = FP/(np.sum(1-labels))
        TPRs.append(TPR)
        FPRs.append(FPR)
    plt.plot(FPRs, TPRs)
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity, Recall)")
    plt.title("ROC plot of real data")
    plt.show()


def data_cleaning(df, test=False):
    ###################### PREPROCESSING ######################
    print "___LOADING DATA___"

    print "___PREPROCESSING___"
    print "- parsing previous payments column"

    print "___building ticket cost variables___"
    # initialize output lists
    cost_mean_out = []
    cost_std_out = []
    q_sold_out = []
    for row in df['ticket_types']:
        # mean of previous payment amounts
        cost_mean_out.append(np.array([entry['cost'] for entry in row]).mean())
        # std deviation of previous payment amounts
        cost_std_out.append(np.array([entry['cost'] for entry in row]).std())
        # number of entries per row -IMPORTANT FEATURE-
        q_sold_out.append(sum([entry['quantity_sold'] for entry in row]))
    df['ticket_avg_cost'] = cost_mean_out
    print "average ticket cost done (ticket_avg_cost)"
    df['ticket_std_cost'] = cost_std_out
    print "stdev ticket cost done (std_cost)"
    df['ticket_q'] = q_sold_out
    print "total ticket quantity done (ticket_q)"

    print "___building previous payment variables___"
    # number of previous payments
    df['prev_pay_len'] = [len(row) for row in df['previous_payouts']]
    print "number of previous payments done (prev_pay_len)"

    # check if venue state is in states included previous payments
    compare = zip(df['venue_state'], df['previous_payouts']) # put venue state alongside list of previous payout states in tuple
    out = []
    for ven_row, pay_row in compare:
        l = [entry['state'] for entry in pay_row] # extract state list from prev pay json
        state_check = (ven_row in l)
        out.append(state_check)
    df['weird_state'] = out * 1

    print "___SPECIALIZED PREPROCESSING DATA___"

    print "___filtering nans___"
    # sale duration 155 nans -> sale duration mean
    df['sale_duration'].fillna(df['sale_duration'].mean(), inplace=True)
    # org_twitter 59 nans -> 0
    df['org_twitter'].fillna(0, inplace=True)
    # org_facebook 59 nans -> 0
    df['org_facebook'].fillna(0, inplace=True)
    df['ticket_avg_cost'] = df['ticket_avg_cost'].fillna(0)
    df['ticket_std_cost'] = df['ticket_std_cost'].fillna(0)


    print "___building dummies___"
    dum_col = ['currency', 'payout_type', 'delivery_method', 'has_header']
    df = change_string_dummies(df, ['currency', 'payout_type'])
    df['has_header'] = df['has_header'].fillna(500)
    df['delivery_method'] = df['delivery_method'].fillna(400)

    if test:
        with open('dummy.pkl') as f:
            enc = pickle.load(f)
    else:
        enc = OneHotEncoder()
        enc.fit(df[dum_col])

        with open('dummy.pkl', 'w') as f:
            pickle.dump(enc, f)
        print "model saved to enc.pkl"

    raw_data = enc.transform(df[dum_col]).toarray()
    dummydf = pd.DataFrame(raw_data)
    df = pd.concat([df, dummydf], axis=1)
    print 'built dummies'


    print '___________________________________________________________'
    print df.columns

    print "___BUILDING MODEL DATAFRAME (VERBOSE)___"
    df_1 = df[['acct_type', 'num_order', 'num_payouts', \
                'has_logo', 'has_analytics',\
                'user_created', 'sale_duration', 'user_age', 'org_twitter', \
                'org_facebook', 'name_length', 'gts', 'body_length', \
                'ticket_avg_cost','ticket_std_cost','ticket_q', 'weird_state', \
                0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]

    print "MODEL DF COLUMNS:"
    print df_1.columns

    print "___generating X, y & feature names___"
    # y = df_1['acct_type'].values
    # X = df_1.drop(['acct_type'], axis=1).values
    return df_1
    # return X, y, df_1.columns

def change_string_dummies(df, dum_string_col):
    i = 0
    mapping = {}
    for col in dum_string_col:
        dummy_columns = df[col].unique()
        for dummy_col in dummy_columns:
            mapping[dummy_col] = i
            i +=1

    df[dum_string_col] = df[dum_string_col].applymap(lambda x: mapping[x])
    return df

if __name__ == '__main__':

    df = pd.read_json("data/train_new.json")
    y_col = 'acct_type'
    df[y_col] = (df[y_col] != 'premium') * 1

    train, test = train_test_split(df, test_size=0.2, random_state=42)
    model = AutomatedClassifier()
    model.fit(train, y_col, oversampling=False)
    pred = model.predict(test)
    model.score(test[y_col], pred)
    cost_benefit = np.array([[0, -50], [-220, 0]])
    model.plot_profit_curve(cost_benefit, test[y_col], ROC=True)

    model = AutomatedClassifier()
    model.fit(train, y_col, oversampling=True, target=0.3)

    # Feature engineer
