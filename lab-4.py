import nltk
from nltk.corpus import wordnet
import random
from sklearn.feature_extraction.text import TfidfVectorizer

# nltk.download('wordnet')
# nltk.download('omw-1.4')

def thesaurus_query_expansion(query):
    tokens = query.lower().split()
    expanded_query = []

    for word in tokens:
        expanded_query.append(word)
        
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym_word = lemma.name().replace('_', ' ')
                expanded_query.append(synonym_word)

    return list(set(expanded_query))


def filter_with_tfidf(words):
    # Convert list to "documents"
    docs = [" ".join(words)]
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(docs)

    scores = zip(vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[0])

    # Sort words by importance
    sorted_words = sorted(scores, key=lambda x: x[1], reverse=True)

    # Take top 10 important words
    top_words = [word for word, score in sorted_words[:10]]

    return top_words


def generate_sentence(word, word_list):
    if len(word_list) >= 3:
        selected = random.sample(word_list, 3)
        return f"The word '{word}' may relate to {selected[0]}, {selected[1]} and {selected[2]}."
    return f"The word '{word}' has multiple meanings."


query = "light"

expanded = thesaurus_query_expansion(query)

print("Expanded Query:", expanded)

# Apply TF-IDF
filtered_words = filter_with_tfidf(expanded)

print("\nTop Words after TF-IDF:", filtered_words)

# Generate sentence
sentence = generate_sentence(query, filtered_words)

print("\nGenerated Sentence:")
print(sentence)