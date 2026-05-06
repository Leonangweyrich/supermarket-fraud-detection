# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

#Packages related to general operating system & warnings
import os
import warnings
warnings.filterwarnings('ignore')
#Packages related to data importing, manipulation, exploratory data #analysis, data understanding
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from termcolor import colored as cl # text customization
#Packages related to data visualizaiton
import seaborn as sns
import matplotlib.pyplot as plt
#matplotlib inline
#Setting plot sizes and type of plot
plt.rc("font", size=14)
plt.rcParams['axes.grid'] = True
plt.figure(figsize=(6,3))
plt.gray()
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.preprocessing import  PolynomialFeatures, KBinsDiscretizer, FunctionTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer, OrdinalEncoder
import statsmodels.formula.api as smf
import statsmodels.tsa as tsa
from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet, Lasso, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
# export
from sklearn.ensemble import BaggingClassifier, BaggingRegressor,RandomForestClassifier,RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.svm import LinearSVC, LinearSVR, SVC, SVR
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    data = pd.read_csv("creditcard.csv")
    Total_transactions = len(data)
    normal = len(data[data.Class == 0])
    fraudulent = len(data[data.Class == 1])
    fraud_percentage = round(fraudulent / normal * 100, 2)
    print(cl('Total number of Trnsactions are {}'.format(Total_transactions), attrs=['bold']))
    print(cl('Number of Normal Transactions are {}'.format(normal), attrs=['bold']))
    print(cl('Number of fraudulent Transactions are {}'.format(fraudulent), attrs=['bold']))
    print(cl('Percentage of fraud Transactions is {}'.format(fraud_percentage), attrs=['bold']))

    min(data.Amount), max(data.Amount)
    sc = StandardScaler()
    amount = data['Amount'].values
    data['Amount'] = sc.fit_transform(amount.reshape(-1, 1))
    data.drop(['Time'], axis=1, inplace=True)
    data.shape
    data.drop_duplicates(inplace=True)
    data.shape

    X = data.drop('Class', axis=1).values
    y = data['Class'].values

    # split our train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

    DT = DecisionTreeClassifier(max_depth=4, criterion='entropy')
    DT.fit(X_train, y_train)
    tree_yhat = DT.predict(X_test)
    print('Accuracy score of the Decision Tree model is {}'.format(accuracy_score(y_test, tree_yhat)))
    print('F1 score of the Decision Tree model is {}'.format(f1_score(y_test, tree_yhat)))
    confusion_matrix(y_test, tree_yhat, labels=[0, 1])

    n = 7
    KNN = KNeighborsClassifier(n_neighbors=n)
    KNN.fit(X_train, y_train)
    knn_yhat = KNN.predict(X_test)
    print('Accuracy score of the K-Nearest Neighbors model is {}'.format(accuracy_score(y_test, knn_yhat)))
    print('F1 score of the K-Nearest Neighbors model is {}'.format(f1_score(y_test, knn_yhat)))

    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    lr_yhat = lr.predict(X_test)
    print('Accuracy score of the Logistic Regression model is {}'.format(accuracy_score(y_test, lr_yhat)))
    print('F1 score of the Logistic Regression model is {}'.format(f1_score(y_test, lr_yhat)))

    svm = SVC()
    svm.fit(X_train, y_train)
    svm_yhat = svm.predict(X_test)
    print('Accuracy score of the Support Vector Machines model is {}'.format(accuracy_score(y_test, svm_yhat)))
    print('F1 score of the Support Vector Machines model is {}'.format(f1_score(y_test, svm_yhat)))

    rf = RandomForestClassifier(max_depth=4)
    rf.fit(X_train, y_train)
    rf_yhat = rf.predict(X_test)
    print('Accuracy score of the Random Forest model is {}'.format(accuracy_score(y_test, rf_yhat)))
    print('F1 score of the Random Forest model is {}'.format(f1_score(y_test, rf_yhat)))

    xgb = XGBClassifier(max_depth=4)
    xgb.fit(X_train, y_train)
    xgb_yhat = xgb.predict(X_test)
    print('Accuracy score of the XGBoost model is {}'.format(accuracy_score(y_test, xgb_yhat)))
    print('F1 score of the XGBoost model is {}'.format(f1_score(y_test, xgb_yhat)))


