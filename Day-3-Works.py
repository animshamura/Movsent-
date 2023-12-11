#train reviews
train_reviews = data.review[:40000]

#test reviews
test_reviews = data.review[40000:]

# train sentiments
train_sentiment = data.review[:40000]

test_sentiment = data.review[40000:]

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
