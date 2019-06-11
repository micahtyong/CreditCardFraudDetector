# coding: utf-8

import sys
import numpy
import pandas
import matplotlib
import seaborn
import scipy
import sklearn

print('Python: {}'.format(sys.version))
print('Numpy: {}'.format(numpy.__version__))
print('Pandas: {}'.format(pandas.__version__))
print('Matplotlib: {}'.format(matplotlib.__version__))
print('Seaborn: {}'.format(seaborn.__version__))
print('Sci Py: {}'.format(scipy.__version__))
print('SK Learn: {}'.format(sklearn.__version__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# LOAD THE DATA SET
data = pd.read_csv('creditcard.csv')

# EXPLORE THE DATA SET
# 1 represents non-fraud
# -1 represents fraude
print(data.columns)

# How many examples do we have?
# m by n matrix where m = number of examples; n = number of features
print(data.shape)

print(data.describe())

# USE ONLY FRACTION OF DATA SET TO SAVE ON COMPUTATION
data = data.sample(frac=0.2, random_state=1)
print(data.shape)

# PLOT HISTOGRAM FOR EACH N PARAMETERS
data.hist(figsize=(20, 20))
plt.show()

# OBSERVATIONS
# Not many fraudulent transations in our data set.
# Most means for features are centered around zero
# AKA Gaussian distributions centered at zero

# Determine fraction of fraud cases in dataset
fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]

outlier_fraction = len(fraud) / float(len(valid))
print(outlier_fraction)

print('Fraud Cases: {}'.format(len(fraud)))
print('Valid Cases: {}'.format(len(valid)))

# Important things to consider: our dataset is skewed

# Correlation matrix (will tell us whether we need to remove specific features or if any are redundant)
corrmat = data.corr()
fig = plt.figure(figsize=(12, 9))

sns.heatmap(corrmat, vmax=0.8, square=True)

# Format dataset; Get all the columns from the DataFrame
columns = data.columns.tolist()

# Filter out the columns to remove data we do not want
columns = [c for c in columns if c not in ['Class']]

# Store the variable we'll be predicting on for our supervised learning algorithm
target = 'Class'

X = data[columns]
Y = data[target]

# Print the shapes of X and Y
print(X.shape)  # m by n features
print(Y.shape)  # m by 1

# APPLYING ML ALGORITHMS TO THE PROJECT
from sklearn.metrics import classification_report, accuracy_score

# We will try the following two algorithms and choose the one that performs best on
# our cross validation data set (which will be derived from the original sample)

# IsolationForest returns the anomaly score by randomly selecting a feature
# and then randomly selecting a split min and max for all examples for that feature
# Number of splits depends on recursive depth.
# Path length is a measure of normality in decision function. In short, isolates random points
# that are likely to be anomalies.
from sklearn.ensemble import IsolationForest

# LocalOutlierFactor is a supervised detection method which returns the
# anomaly score of each example in sample. More specifically, how far
# an example deviates from it's k nearest neighbors.
from sklearn.neighbors import LocalOutlierFactor

# Define a random state
state = 1

# Define the outlier detection methods; store into Python dictionary
classifiers = {
    "Isolation Forest": IsolationForest(max_samples=len(X),
                                        contamination=outlier_fraction,
                                        random_state=state),
    "Local Outlier Factor": LocalOutlierFactor(
        n_neighbors=12,  # can be varied
        contamination=outlier_fraction)
}

# Fit our model
n_outliers = len(fraud)

for i, (clf_name, clf) in enumerate(classifiers.items()):
    # Fit the data and tag outliers
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    elif clf_name == "Isolation Forest":
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)

    # Y_pred will be a +1 for non-anomaly and -1 for fraud
    # Thus, we must reshape the prediction values to 0 for valid, 1 for fraud
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1

    n_errors = (y_pred != Y).sum()

    # Run classification metrics
    print('{}: {}'.format(clf_name, n_errors))
    print(accuracy_score(Y, y_pred))
    print(classification_report(Y, y_pred))

# Observations

# Local Outlier Factor: Precission and Recall for fraudulent credit cards are too small
# We have way too many false positives and false negatives. F1 score will be a bit low (0.01)

# Isolation Forest isn't too great as well. F1 score is (0.3)
# A bit too many false positives and false negatives.

# Conclusion: Isolation Forest is the better of the two.

