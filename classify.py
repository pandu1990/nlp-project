import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import nltk

data = pd.read_csv('cleaned_data.csv')
#data.head(10)


vec = TfidfVectorizer()
tfidf = vec.fit_transform(data.synopsis).toarray()
train_x = tfidf[:2565]
test_x = tfidf[2565:]
train_y = data.label[:2565]
test_y = data.label[2565:]

#naive bayes
clf = MultinomialNB().fit(train_x, train_y)
predicted = clf.predict(test_x)
print("naive bayes = " + str(np.mean(predicted == test_y)))


#svm

clf_svm = SGDClassifier(loss='hinge',penalty='l2',alpha=1e-3, n_iter=5,random_state=42)
_ = clf_svm.fit(train_x, train_y)
svm_pred = clf_svm.predict(test_x)
print("svm = " + str(np.mean(svm_pred == test_y)))

#Logistic Regression

clf_logreg=LogisticRegression().fit(train_x, train_y)
predicted = clf_logreg.predict(test_x)
print("Logistic Regression = " + str(np.mean(predicted == test_y)))

#Decision Tree

clf_dec = tree.DecisionTreeClassifier(max_depth=7).fit(train_x, train_y)
predicted = clf_dec.predict(test_x)
print("Decision Tree = " + str(np.mean(predicted == test_y)))

#Random Forest

clf_random = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=0).fit(train_x, train_y)
predicted = clf_random.predict(test_x)
print("Random Forest = " + str(np.mean(predicted == test_y)))

#k nearest neighbor

clf_knn = KNeighborsClassifier(n_neighbors=7).fit(train_x, train_y)
predicted = clf_knn.predict(test_x)
print("k Nearest Neighbors = " + str(np.mean(predicted == test_y)))
