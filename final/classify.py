import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

import nltk


new_names = ['title','synopsis','label']
data = pd.read_csv("clean_dataset.tsv", skiprows=1, sep='\t',names=new_names)
#data.head(10)


vec = TfidfVectorizer()
tfidf = vec.fit_transform(data.synopsis).toarray()
train_x = tfidf[:2852]
test_x = tfidf[2852:]
train_y = data.label[:2852]
test_y = data.label[2852:]
#naive bayes
clf = MultinomialNB().fit(train_x, train_y)
predicted = clf.predict(test_x)
print("naive bayes = " + str(np.mean(predicted == test_y)))
f1 = f1_score(test_y, predicted,pos_label="Mature")
print("F1: ",f1)


#svm

clf_svm = SGDClassifier(loss='hinge',penalty='l2',alpha=1e-3, n_iter=5,random_state=42)
_ = clf_svm.fit(train_x, train_y)
svm_pred = clf_svm.predict(test_x)
print("svm = " + str(np.mean(svm_pred == test_y)))
f1 = f1_score(test_y, svm_pred,pos_label="Mature")
print("F1: ",f1)

#Logistic Regression

clf_logreg=LogisticRegression().fit(train_x, train_y)
predicted = clf_logreg.predict(test_x)
print("Logistic Regression = " + str(np.mean(predicted == test_y)))
f1 = f1_score(test_y, predicted,pos_label="Mature")
print("F1: ",f1)

#Decision Tree

clf_dec = tree.DecisionTreeClassifier(max_depth=7).fit(train_x, train_y)
predicted = clf_dec.predict(test_x)
print("Decision Tree = " + str(np.mean(predicted == test_y)))
f1 = f1_score(test_y, predicted,pos_label="Mature")
print("F1: ",f1)

#Random Forest

clf_random = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=0).fit(train_x, train_y)
predicted = clf_random.predict(test_x)
print("Random Forest = " + str(np.mean(predicted == test_y)))
f1 = f1_score(test_y, predicted,pos_label="Mature")
print("F1: ",f1)

#k nearest neighbor

clf_knn = KNeighborsClassifier(n_neighbors=7).fit(train_x, train_y)
predicted = clf_knn.predict(test_x)
print("k Nearest Neighbors = " + str(np.mean(predicted == test_y)))
f1 = f1_score(test_y, predicted,pos_label="Mature")
print("F1: ",f1)


