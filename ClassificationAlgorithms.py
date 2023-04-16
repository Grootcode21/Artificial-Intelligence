import pandas as pd
import matplotlib.pyplot as plt

fruits = pd.read_table('C://Users/ALIENWARE//fruit_data.txt')
print(fruits.head())

print(fruits.shape)
print(fruits['fruit_name'].unique())
print(fruits.groupby('fruit_name').size())

import seaborn as sns

sns.countplot(fruits['fruit_name'], label='Count')
plt.show()

fruits.drop('fruit_label', axis=1).plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False,
                                        figsize=(9, 9),
                                        title='Box plot for each input variable')
plt.savefig('fruits_box')
plt.show()

import pylab as pl

fruits.drop('fruit_label', axis=1).hist(bins=30, figsize=(9, 9))
pl.suptitle('Histogram for each numeric input variable')
plt.savefig(fruits_hist)
plt.show()

feature_names = ['mass', 'width', 'height', 'color_score']
x = fruits[feature_names]
y = fruits['fruit_label']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

# normalize data to avoid bias
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Now we build a model
# Logistic Regression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_trian, y_train)
print('Accuracy of Logistic regression classifier on training set: {:.2f}'
      .format(logreg.score(x_train, y_train)))
print('Accuracy of Logistic Regression classifier on test set: {:.2f}'
      .format(logreg.score(x_test, y_test)))

# Decision Tree
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier().fit(x_trian, y_train)
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
      .format(clf.score(x_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
      .format(clf.score(x_test, y_test)))

# KNN Classifier
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_trian, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'
      .format(knn.score(x_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
      .format(knn.score(x_test, y_test)))

# NaiveBayes
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(x_trian, y_train)
print('Accuracy of GNB classifier on training set: {:.2f}'
      .format(gnb.score(x_train, y_train)))
print('Accuracy of GNB classifier on test set: {:.2f}'
      .format(gnb.score(x_test, y_test)))

# Support Vector Machine Classifier
from sklearn.svm import SVC

svm = SVC()
svm.fit(x_trian, y_train)
print('Accuracy of SVM classifier on training set: {:.2f}'
      .format(svm.score(x_train, y_train)))
print('Accuracy of SVM classifier on test set: {:.2f}'
      .format(svm.score(x_test, y_test)))

# Since the KNN Classifier classified our data more accurately, we look at what it predicts
gamma = 'auto'
# Confusion matrix for knn classifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

pred = knn.predict(x_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
