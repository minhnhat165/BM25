from math import log

import nltk
from nltk.corpus import stopwords

# nltk.download('punkt') # uncomment if you haven't downloaded the package
# nltk.download('stopwords') # uncomment if you haven't downloaded the package

# Preprocessing data
TOP_RESULT = 3
k = 2
b = 0.75
idf = {}

# preprocessing data


def remove_stop_words(doc):
    doc = doc.lower()
    return ' '.join([word for word in doc.split() if word not in stopwords.words('english')])


def preprocess_query(query):
    query = remove_stop_words(query)
    # return a set of query
    return set(query.split())


def preprocess_docs(docs):
    removed_stopwords_docs = []
    for doc in docs:
        removed_stopwords_docs.append(remove_stop_words(doc))
    # return a list of docs after removing stop words, number of docs, average length of docs
    return removed_stopwords_docs, len(removed_stopwords_docs), compute_average_length(removed_stopwords_docs)


def TF(term, doc):
    return doc.split().count(term)


def DF(term, docs):
    i = 0
    for doc in docs:
        if term in doc:
            i += 1
    return i


def IDF(term, N, docs):
    df = DF(term, docs)
    if df == 0:
        log(N / 1)
    return log(N / df)


def compute_average_length(docs):
    sum = 0
    for doc in docs:
        sum += len(doc.split())
    return sum / len(docs)


def weight_BM25(term, doc, N, avgdl, docs):
    global k
    global b
    global idf
    if term not in idf:
        idf[term] = IDF(term, N, docs)
    tf = TF(term, doc)
    dl = len(doc.split())
    rsv = idf[term] * ((k + 1) * tf) / (k * (1 - b + b * (dl / avgdl)) + tf)
    return rsv


def RSV_BM25(query, doc, N, avgdl, docs):
    sum = 0
    for term in query:
        sum += weight_BM25(term, doc, N, avgdl, docs)
    return sum


def rank_BM25(query, docs, N, avgdl):
    query_set = preprocess_query(query)
    rsv = []
    for i, doc in enumerate(docs):
        rsv.append((i, RSV_BM25(query_set, doc, N, avgdl, docs)))
    sorted_rsv = sorted(rsv, key=lambda x: x[1], reverse=True)
    return sorted_rsv


if __name__ == "__main__":
    # input data
    docs = [
        'Human machine interface for lab abc computer applications',
        'A survey of user opinion of computer system response time',
        'The EPS user interface management system',
        'System and human system engineering testing of EPS',
        'Relation of user perceived response time to error measurement',
        'The generation of random binary unordered trees',
        'The intersection graph of paths in trees',
        'Graph minors IV Widths of trees and well quasi ordering',
        'Graph minors A survey'
    ]
    query = 'The intersection of graph survey and trees trees survey'

    processed_docs, N, avgdl = preprocess_docs(docs)

    rank = rank_BM25(query, processed_docs, N, avgdl)

    top_index = []

    for i, rsv in rank[:TOP_RESULT]:
        print("doc", i + 1, ":", rsv)
        print(docs[i])
        print("------------------------------------------------")
        top_index.append(i+1)
    print("------------------------------------------------")
    print("Top", TOP_RESULT, ":", top_index)
