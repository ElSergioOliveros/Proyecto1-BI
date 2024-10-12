from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, min_df=0.017):
        nltk.download('stopwords')
        nltk.download('punkt')
        self.spanish_stopwords = set(stopwords.words('spanish'))
        self.spanish_stopwords.add("ello")
        self.regexStopWords = r'\b(' + '|'.join(self.spanish_stopwords) + r')\b'

        self.stemmer = SnowballStemmer("spanish")
        self.tfidfVectorizer = TfidfVectorizer(min_df=min_df)

    def fit(self, textsDf, y=None):
        textsDf = textsDf.str.replace(self.regexStopWords, "", regex=True)
        textsDf = textsDf.str.replace(r'[^\w\s]', '', regex=True)
        textsDf = textsDf.str.replace(r'\d+|_', '', regex=True)
        
        stems = textsDf.apply(lambda text : [self.stemmer.stem(token) for token in word_tokenize(text)])

        self.tfidfVectorizer.fit(stems.astype(str))
        
        return self

    def transform(self, textsDf):
        textsDf = textsDf.str.replace(self.regexStopWords, "", regex=True)
        textsDf = textsDf.str.replace(r'[^\w\s]', '', regex=True)
        textsDf = textsDf.str.replace(r'\d+|_', '', regex=True)
        
        stems = textsDf.apply(lambda text : [self.stemmer.stem(token) for token in word_tokenize(text)])
        tfidfDocumentsMat = self.tfidfVectorizer.transform(stems.astype(str)).toarray()

        return tfidfDocumentsMat
    
    def fit_transform(self, textsDf, y=None):
        return self.fit(textsDf).transform(textsDf)


        
