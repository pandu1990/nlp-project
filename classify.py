import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

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