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
     
