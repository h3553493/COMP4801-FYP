import csv
import json
import re
from nltk import pos_tag
import nltk.corpus
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords, wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from datetime import datetime

def process_corpus(Corpus):
    name=Corpus['Company Name']; ticker=Corpus['Ticker']; cik=Corpus['CIK']; formtype=Corpus['formType']; year=Corpus['Filing Year']; isFraud=Corpus['isFraud']; text=Corpus['MD&A']
    
    #turn capitalization to lowercase
    Corpus['MD&A']=[text.lower() for text in Corpus['MD&A']]

    #remove unicode characters
    Corpus['MD&A'] =[re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", text) for text in Corpus['MD&A']]
    Corpus['MD&A']=[" ".join(text.split()) for text in Corpus['MD&A']]
    
    #remove digits
    Corpus['MD&A']=[re.sub(r'[0-9]','',text) for text in Corpus['MD&A']]

    #remove stopwords
    stopword=stopwords.words('english')
    Corpus['MD&A']=[" ".join([word for word in text.split() if word not in (stopword)]) for text in Corpus['MD&A']]

    #Tokenization
    Corpus['MD&A']=[word_tokenize(word) for word in Corpus['MD&A']]

    #Word stemming/Lemmenting
    tag_map = defaultdict(lambda : wn.NOUN); tag_map['J'] = wn.ADJ; tag_map['V'] = wn.VERB; tag_map['R'] = wn.ADV
    for index,words in enumerate(Corpus['MD&A']):
        Final_words = []
        # Initializing WordNetLemmatizer()
        word_Lemmatized = WordNetLemmatizer()
        # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
        for word, tag in pos_tag(words):
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
        # The final processed set of words for each iteration will be stored in 'text_final'
        Corpus.loc[index,'Tokenized_text']=str(Final_words)
    return Corpus