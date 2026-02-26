import re
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# nltk.download('punkt')
# nltk.download('stopwords')

documents = [
    "Stanford University is a great place for research. Many students aspire to study here.",
    "California is known for its beautiful weather and top-ranked universities.",
    "The University of California, Berkeley, is also a highly reputed institution.",
    "MIT is one of the most prestigious universities in the world.",
    "Harvard University has a long history of excellence in education and research."
]

documents2 = [
    "Artificial Intelligence is transforming industries worldwide.",
    "Machine Learning and Deep Learning are subsets of AI.",
    "The impact of AI on the healthcare sector is revolutionary.",
    "AI technologies are being used in autonomous vehicles.",
    "Ethical concerns around AI are growing as the technology advances."
]

documents3 = [
    "The University of California, Berkeley, is also a highly reputed institution."
]

documents4 = [
    "MIT is one of the most prestigious universities in the world."
]

documents5 = [
    "Harvard University has a long history of excellence in education and research."
]

all_documents = documents + documents2 + documents3 + documents4 + documents5

query = "AI in health care"

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [stemmer.stem(w) for w in tokens]
    return " ".join(tokens)

processed_docs = [preprocess(doc) for doc in all_documents]
processed_query = preprocess(query)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(processed_docs)
query_vector = vectorizer.transform([processed_query])

similarity = cosine_similarity(query_vector, tfidf_matrix)
scores = similarity.flatten()

ranked_indices = np.argsort(scores)[::-1]

print("Query:", query)
print("\nDocument Ranking:\n")

for i in ranked_indices:
    print("Score:", round(scores[i], 4))
    print("Document:", all_documents[i])
    print("-" * 60)