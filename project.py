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

import nltk
nltk.download('punkt')
data['review'] = data['review'].apply(word_tokenize)

data['review'][1]

data['review'] = data['review'].apply(lambda x: [item for item in x if item.isalpha()])
     
print(" ".join(data['review'][1]))
     
data['review'] = data['review'].apply(lambda x: [item.lower() for item in x])
     
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

data['review'] = data['review'].apply(lambda x: [item for item in x if item not in stop_words])

from nltk.stem import PorterStemmer
ps = PorterStemmer()
data['review'] = data['review'].apply(lambda x: [ps.stem(item) for item in x])

data['review'] = data['review'].apply(lambda x: " ".join(x))

#train reviews
train_reviews = data.review[:40000]

#test reviews
test_reviews = data.review[40000:]

# train sentiments
train_sentiment = data.sentiment[:40000]

test_sentiment = data.sentiment[40000:]

#Count vectorizer for bag of words
cv = CountVectorizer(min_df=0, max_df=1, binary = False, ngram_range = (1,3))

#transformed train reviews
cv_train_reviews = cv.fit_transform(train_reviews)

#transformed test reviews
cv_test_reviews = cv.transform(test_reviews)

#labeling the sentient data
lb = LabelBinarizer()

# transformed sentiment data
lb_train_sentiments = lb.fit_transform(train_sentiment)

# transformed test sentiment data (similar to count vectorizer, transform test reviews, name it lb_test_sentiments)
lb_test_binary = lb.fit_transform(test_sentiment)

# training the model
mnb = MultinomialNB()

# fitting the model
mnb_bow = mnb.fit(cv_train_reviews, lb_train_sentiments)

# predicting the model for bag of words
mnb_bow_predict = mnb.predict(cv_test_reviews)

# accuracy score for bag of words
mnb_bow_score = accuracy_score(lb_test_binary, mnb_bow_predict)
print("Accuracy :", mnb_bow_score)

# commented out IPython magic to ensure Python compatibility.
# word cloud for positive review words in the entire dataset
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
# %matplotlib inline

# join all the positive reviews
positive_words = ' '.join(list(data[data['sentiment'] == 'positive']['review']))

# word cloud for positive words
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(positive_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()
