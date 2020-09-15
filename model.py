import nltk 
from nltk.tokenize import word_tokenize
import re 
import numpy as np 
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import pickle

posts=pd.read_csv('C:\\Users\\DELL\\Desktop\\facebook_posrs.csv', encoding = "cp1252")

def simple_split(data,y,length,split_mark=0.7):
    if split_mark > 0. and split_mark < 1.0:
        n = int(split_mark*length)
    else:
        n = int(split_mark)
    X_train = data [:n].copy()
    X_test = data[n:].copy()
    y_train = y[:n].copy()
    y_test = y[n:].copy()
    return X_train,X_test,y_train,y_test

vect = CountVectorizer()

X_train,X_test,y_train,y_test = simple_split(posts.message,posts.label,len(posts))

#Learn all the vocabulary words from X_train and apply tranformation to build the bag of words on X_test as well as transform the Train
X_train = vect.fit_transform(X_train)
X_test = vect.transform(X_test)

message = " I am feeling happy"

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

pickle.dump(logreg, open('logreg.pkl','wb'))
model1 = pickle.load(open('logreg.pkl','rb'))
print(model1.predict(vect.transform([message]))[0])

nb = MultinomialNB()
nb.fit(X_train, y_train)

pickle.dump(nb, open('nb.pkl','wb'))
model2 = pickle.load(open('nb.pkl','rb'))
print(model2.predict(vect.transform([message]))[0])

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

pickle.dump(rf, open('rf.pkl','wb'))
model3 = pickle.load(open('rf.pkl','rb'))
print(model3.predict(vect.transform([message]))[0])
