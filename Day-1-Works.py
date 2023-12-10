#importing necessary libraries 

import pandas as pd 
import numpy as nm
import matplotlib.pyplot as mp
import nltk.tokenize as nt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import seaborn as sb


#Setting url and data

url = "https://raw.githubusercontent.com/meghjoshii/NSDC_DataScienceProjects_SentimentAnalysis/main/IMDB%20Dataset.csv"
data = pd.read_csv(url)

#Getting top 5 rows of the dataframe

data.head()

#Describing data

data.describe()

#Printing reviews

print(data['review'])

#Printing sentiments

sentiment_count = data['sentiment'].value_counts()
print(sentiment_count)



