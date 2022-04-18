import csv
import json
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from datetime import datetime
from preprocessing_func import process_corpus


#Load the corpus and do the preprocessing
print("Loading Corpus and do the preprocessing")
Corpus=pd.read_json('./data/corpus-large.json')
Corpus=process_corpus(Corpus)
print("Successfully Loaded the corpus")

#obtain the training and testing dataset
Train_X, Test_X, Train_Y, Test_Y=model_selection.train_test_split(Corpus['Tokenized_text'],Corpus['isFraud'],test_size=0.3)

#Encoding the dataset
Encoder=LabelEncoder()
Train_Y=Encoder.fit_transform(Train_Y)
Test_Y=Encoder.fit_transform(Test_Y)

#Word Vectorization using TF-IDF
Tfidf_vect=TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(Corpus['Tokenized_text'])
Train_X_Tfidf=Tfidf_vect.transform(Train_X)
Test_X_Tfidf=Tfidf_vect.transform(Test_X)

#Using Naive Bayes Classifier to predict the outcome
Naive=naive_bayes.MultinomialNB()
Naive.fit(Train_X_BOW,Train_Y)

predictions_NB=Naive.predict(Test_X_BOW)
print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)
print("Confusion matrix is")
print(confusion_matrix(predictions_NB, Test_Y))
print("classification report is")
print(classification_report(predictions_NB, Test_Y))

#Using SVM to predict the outcome
SVM=svm.SVC(C=1.0,kernel='linear',degree=3,gamma='auto')
SVM.fit(Train_X_BOW,Train_Y)

predictions_SVM=SVM.predict(Test_X_BOW)
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)
print("Confusion matrix is")
print(confusion_matrix(predictions_SVM, Test_Y))
print("classification report is")
print(classification_report(predictions_SVM, Test_Y))

#Using Linear Regression to predict the outcome
LR=LogisticRegression(C=100, random_state=0, max_iter=1000)
LR.fit(Train_X_BOW,Train_Y)
predictions_LR=LR.predict(Test_X_BOW)
print("LR Accuracy Score -> ",accuracy_score(predictions_LR, Test_Y)*100)
print("Confusion matrix is")
print(confusion_matrix(predictions_LR, Test_Y))
print("classification report is")
print(classification_report(predictions_LR, Test_Y))
