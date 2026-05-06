import os
import warnings

warnings.filterwarnings('ignore')
# Packages related to data importing, manipulation, exploratory data #analysis, data understanding
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from termcolor import colored as cl  # text customization
# Packages related to data visualizaiton
import seaborn as sns
import matplotlib.pyplot as plt

# matplotlib inline
# Setting plot sizes and type of plot
plt.rc("font", size=14)
plt.rcParams['axes.grid'] = True
plt.figure(figsize=(6, 3))
plt.gray()
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.preprocessing import PolynomialFeatures, KBinsDiscretizer, FunctionTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer, OrdinalEncoder
import statsmodels.formula.api as smf
import statsmodels.tsa as tsa
from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet, Lasso, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
# export
from sklearn.ensemble import BaggingClassifier, BaggingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, \
    AdaBoostRegressor
from sklearn.svm import LinearSVC, LinearSVR, SVC, SVR
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from typing import Tuple
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA

Q_OUTLIERS = 50
Q_DATAPOINTS = 200


def count_anomalies(li: np.ndarray) -> int:
    return sum(map(lambda x: x == -1, li))


def create_dataset(q_train: int, q_out: int) -> Tuple[np.ndarray, np.ndarray]:
    X = 0.3 * np.random.randn(q_train, 2)
    X_norm = np.r_[X + 2, X - 2]

    X_outliers = np.random.uniform(low=-4, high=4, size=(q_out, 2))

    return X_norm, X_outliers


def correlate_indexes(pred_arr: np.ndarray, original_arr: np.ndarray) -> np.ndarray:
    # Get outliers' index
    anom_index = np.where(pred_arr == -1)

    # Rerturn outliers' exact position from base set by correlating indexes
    return original_arr[anom_index]


def metrics_gen(orig_outliers: np.ndarray, predicted_outliers: np.ndarray,
                correct_outliers: np.ndarray, false_outliers: np.ndarray,
                undetected_outliers: np.ndarray):
    print('Accuracy on test set: {:.2f}%'.format(100 * len(correct_outliers) / len(X_test_outliers)))
    print('False detection ratio: {:.2f}%'.format(100 * len(false_outliers) / len(test_anom)))
    print('Undetection ratio: {:.2f}%'.format(100 * len(undetected_outliers) / len(X_test_outliers)))


def map1to0(n):
    if n == 1:
        return 0
    else:
        return 1


if __name__ == '__main__':

    data = pd.read_csv("supermarket_1.csv", usecols=[0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], header=None)
#                     usecols=[0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
#                             25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
#                             47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
#                             69, 70, 71, 72, 73, 74], header=None)
    print(data)

    Total_transactions = len(data)
    print(cl('Total number of Trnsactions are {}'.format(Total_transactions), attrs=['bold']))

    pca = PCA(0.9)
    pca.fit(data)

    min(data.Amount), max(data.Amount)
    sc = StandardScaler()
    amount = data['Amount'].values
    data['Amount'] = sc.fit_transform(amount.reshape(-1, 1))
    data.shape

    X = data.drop('Class', axis=1).values
    y = data['Class'].values

    # split our train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

    plt.figure(figsize=(15, 7.5))
    plt.title("Dataset distribution")

    plt.scatter(X_train[:, 0], X_train[:, 1], label='X_train')
    plt.scatter(X_test[:, 0], X_test[:, 1], label='X_test_normal')
    plt.legend()
    plt.show()

    print("step 1")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7.5))

    # Plot train set
    ax1.set_title('Train set distribution')
    ax1.scatter(X_train[:, 0], X_train[:, 1], color='indigo')

    print("step 2")

    # Legend for train set
    legend_elements = [Line2D([], [], marker='o', label='Normal datapoint', color='indigo', linestyle='None'),
                       Line2D([], [], marker='o', label='Outlier', color='red', linestyle='None')]
    ax1.legend(handles=legend_elements)

    print("step 3")

    # Plot test set
    ax2.set_title('Test set distribution')
    ax2.scatter(X_test[:, 0], X_test[:, 1], color='indigo')

    print("step 4")

    # Legend for test set
    legend_elements = [Line2D([], [], marker='o', label='Normal datapoint', color='indigo', linestyle='None'),
                       Line2D([], [], marker='o', label='Outlier', color='red', linestyle='None')]
    ax2.legend(handles=legend_elements)
    print("step 5")
    plt.show()

    print("step 6")

    clf = IsolationForest(random_state=42, contamination=0.1)
    clf.fit(X_train)

    print("step 7")

    y1_train = clf.predict(X_train)
    y1_test = clf.predict(X_test)

    print('Input Train Class')
    for i in range(0, 100):
        print(y_train[i]),

    print('Detected Train Class')
    for i in range(0, 100):
        print(y1_train[i]),

    #    y1m_train = map(map1to0, y1_train)
    #    y1m_test = map(map1to0, y1_test)
    y1m_train = clf.predict(X_train)
    y1m_test = clf.predict(X_test)

    for i in range(0, len(y1_train)):
        y1m_train[i] = map1to0(y1_train[i])

    for i in range(0, len(y1_test)):
        y1m_test[i] = map1to0(y1_test[i])

    print('Detected mTrain Class')
    for i in range(0, 100):
        print(y1m_train[i]),

    correct_train = np.array([val for val in y_train if val in y1_train])
    correct_test = np.array([val for val in y_test if val in y1_test])
    print('Correct train detections: {}'.format(len(correct_train)))
    print('Correct test detections: {}'.format(len(correct_test)))

    correctm_train = np.array([val for val in y_train if val in y1m_train])
    correctm_test = np.array([val for val in y_test if val in y1m_test])
    print('Correct mtrain detections: {}'.format(len(correctm_train)))
    print('Correct mtest detections: {}'.format(len(correctm_test)))

    correctoutm_train = np.array([val for val in y_train if (val == 1 and val in y1m_train)])
    correctoutm_test = np.array([val for val in y_test if (val == 1 and val in y1m_test)])
    print('Correct Out mtrain detections: {}'.format(len(correctoutm_train)))
    print('Correct Out mtest detections: {}'.format(len(correctoutm_test)))

    train_anom = correlate_indexes(y1_train, X_train)
    test_anom = correlate_indexes(y1_test, X_test)

    plt.figure(figsize=(15, 7.5))
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y1_train)
    plt.title('Predictions on training set')

    # This is to set the legend appropriately
    legend_elements = [Line2D([], [], marker='o', color='yellow', label='Marked as normal', linestyle='None'),
                       Line2D([], [], marker='o', color='indigo', label='Marked as anomaly', linestyle='None')]
    plt.legend(handles=legend_elements)
    plt.show()

    plt.figure(figsize=(15, 7.5))
    plt.title('Predictions on test set')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y1_test)

    legend_elements = [Line2D([], [], marker='o', color='yellow', label='Marked as normal', linestyle='None'),
                       Line2D([], [], marker='o', color='indigo', label='Marked as anomaly', linestyle='None')]
    plt.legend(handles=legend_elements)
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7.5))
