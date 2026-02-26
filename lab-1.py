import urllib.request
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from bs4 import BeautifulSoup

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

url = "https://www.geeksforgeeks.org/machine-learning/machine-learning/"

# html = urllib.request.urlopen(url).read().decode("utf-8").lower()
# print(html)
# # text = re.sub(r'<.*?>', ' ', html)
# # tokens = word_tokenize(text)

# # stop_words = set(stopwords.words("english"))
# # filtered = [w for w in tokens if w.isalpha() and w not in stop_words]

# # stemmer = PorterStemmer()
# # stemmed = [stemmer.stem(w) for w in filtered]

# # lemmatizer = WordNetLemmatizer()
# # lemmatized = [lemmatizer.lemmatize(w) for w in filtered]

# # print("after tokenization",tokens[:20])


html = urllib.request.urlopen(url).read()
soup = BeautifulSoup(html, "html.parser")

text = soup.get_text().lower()

tokens = word_tokenize(text)

stop_words = set(stopwords.words("english"))
filtered_words = [w for w in tokens if w.isalpha() and w not in stop_words]

stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(w) for w in filtered_words]

lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(w) for w in filtered_words]

print("Tokens:\n", tokens)
print("after stopwords removal:\n", filtered_words)
print("after stemming:\n",stemmed_words)
print("after lemmatization:",lemmatized_words)