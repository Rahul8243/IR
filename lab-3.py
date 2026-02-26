import re

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    # Tokenize into words
    words = text.split()
    
    return words


def build_inverted_index(documents):
    inverted_index = {}
    
    for doc_id, content in documents.items():
        words = preprocess_text(content)
        
        for word in words:
            if word not in inverted_index:
                inverted_index[word] = set()
            
            inverted_index[word].add(doc_id)
    
    return inverted_index


def boolean_query(query, inverted_index):
    tokens = query.lower().split()
    
    result = set()
    operator = None
    
    for token in tokens:
        if token in ["and", "or", "not"]:
            operator = token
        else:
            docs = inverted_index.get(token, set())
            
            if not result:
                result = docs
            else:
                if operator == "and":
                    result = result.intersection(docs)
                elif operator == "or":
                    result = result.union(docs)
                elif operator == "not":
                    result = result.difference(docs)
    
    return result


documents = {
    1: "The quick brown fox jumps over the lazy dog.",
    2: "The lazy dog lies in the sun.",
    3: "A fox is quick and clever.",
    4: "Sunshine makes the fox happy."
}


inverted_index = build_inverted_index(documents)

# Display inverted index
print("Inverted Index:")
for word, doc_ids in inverted_index.items():
    print(word, ":", doc_ids)

print("\nQuery Results:")
print("fox AND quick:", boolean_query("fox AND quick", inverted_index))
print("fox AND NOT lazy:", boolean_query("fox AND NOT lazy", inverted_index))
print("sun OR happy:", boolean_query("sun OR happy", inverted_index))