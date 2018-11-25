from time import time
from random import shuffle
import pandas as pd
import os
from nltk.corpus import stopwords
import string
import re
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from collections import Counter
import math
import csv
def rem_sw(df):
    # Downloading stop words
    stop_words = set(stopwords.words('english'))


    # Removing Stop words from training data
    count = 0
    for sentence in df:
        sentence = [word for word in sentence.lower().split() if word not in stop_words]
        sentence = ' '.join(sentence)
        df.loc[count] = sentence
        count+=1
    return(df)

def rem_punc(df):
	count = 0
	for s in df:
		cleanr = re.compile('<.*?>')
		s = re.sub(r'\d+', '', s)
		s = re.sub(cleanr, '', s)
		s = re.sub("'", '', s)
		s = re.sub(r'\W+', ' ', s)
		s = s.replace('_', '')
		df.loc[count] = s
		count+=1	
	return(df)

def lemma(df):

    lmtzr = WordNetLemmatizer()

    count = 0
    stemmed = []
    for sentence in df:    
        word_tokens = word_tokenize(sentence)
        for word in word_tokens:
            stemmed.append(lmtzr.lemmatize(word))
        sentence = ' '.join(stemmed)
        df.iloc[count] = sentence
        count+=1
        stemmed = []
    return(df)

def stemma(df):

    stemmer = SnowballStemmer("english") #SnowballStemmer("english", ignore_stopwords=True)

    count = 0
    stemmed = []
    for sentence in df:
        word_tokens = word_tokenize(sentence)
        for word in word_tokens:
            stemmed.append(stemmer.stem(word))
        sentence = ' '.join(stemmed)
        df.iloc[count] = sentence
        count+=1
        stemmed = []
    return(df)
new_names = ['title','synopsis','label']
df = pd.read_csv("dataset.tsv", skiprows=1, sep='\t',names=new_names)
df['synopsis'] = rem_punc(df['synopsis'])
df['synopsis'] = rem_sw(df['synopsis'])
df['synopsis'] = lemma(df['synopsis'])
df['synopsis'] = stemma(df['synopsis'])

f = open('clean_dataset.tsv', 'w')

for i in range(0,len(df['title'])):
    f.write(df['title'][i])
    f.write("\t")
    f.write(df['synopsis'][i])
    f.write("\t")
    f.write(df['label'][i])
    f.write("\n")
f.close()

