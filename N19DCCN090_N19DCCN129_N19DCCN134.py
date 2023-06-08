# Nhóm N19DCCN090
# Nguyễn Minh Nhật - N19DCCN129
# Trần Thị Kim Oanh - N19DCCN134
# Phạm Văn Khánh - N19DCCN090


from math import log, log10

import nltk
from nltk.corpus import stopwords

# nltk.download('punkt') # uncomment if you haven't downloaded the package
# nltk.download('stopwords') # uncomment if you haven't downloaded the package

## Preprocessing data
DATA_FOLDER = "data"
TOP_K = 3
k = 2
b = 0.75
idf = {}

def generate_data_path(file_name):
    return DATA_FOLDER + "/" + file_name


def read_file(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()

    docs = []
    doc = ''
    for line in lines:
        line = line.strip()
        if line == '/':
            docs.append(doc.strip())
            doc = ''
        elif line[0].isdigit():
            continue
        else:
            if doc and not doc[-1].isalnum() and not line[0].isalnum() and not line[0].isspace():
                doc = doc.rstrip() + line.lstrip()
            else:
                doc += ' ' + line

    return docs


def write_file(file_name, docs):
    with open(file_name, "w") as file:
        for i, sublist in enumerate(docs):
            file.write(str(i+1) + "\n")
            for element in sublist:
                file.write(" " + str(element))
            file.write("\n/\n")



# tiền xử lý dữ liệu
def remove_stop_words(doc):
    doc = doc.lower()
    return ' '.join([word for word in doc.split() if word not in stopwords.words('english')])

def tokenize_doc(doc):
    words = nltk.word_tokenize(doc)
    words = [word.lower() for word in words if word.lower(
    ) not in stopwords.words('english') and word.isalnum()]

    return words

# ## Precomputing weights
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
        return 0
    return log10(N / df)
    

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
    for term in query.split():
        sum += weight_BM25(term, doc, N, avgdl, docs)
    return sum
def rank_BM25(query, docs, N, avgdl):
    rsv = []
    for i, doc in enumerate(docs):
        rsv.append((i, RSV_BM25(query, doc, N, avgdl, docs)))
    sorted_rsv = sorted(rsv, key=lambda x: x[1], reverse=True)
    return sorted_rsv


if __name__ == "__main__":
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
    query = 'The intersection of graph survey and trees'
    removed_stopwords_docs = []
    for doc in docs:
        removed_stopwords_docs.append(remove_stop_words(doc))
    N = len(removed_stopwords_docs)
    avgdl = compute_average_length(removed_stopwords_docs)
    
    removed_stopwords_query = remove_stop_words(query)

    rank = rank_BM25(removed_stopwords_query, removed_stopwords_docs, N, avgdl)
    top_index = []
    for i, rsv in rank[:TOP_K]:
        print("doc", i + 1, ":", rsv)
        print(docs[i])
        top_index.append(i+1)
    print("------------------------------------------------")
    print("Top ", TOP_K, ": ", top_index)
    

    # docs = read_file(generate_data_path("doc-text"))
    # removed_stopwords_docs = []
    # for doc in docs:
    #     print(doc)
    #     removed_stopwords_docs.append(remove_stop_words(doc).lower())
    # N = len(removed_stopwords_docs)
    # avgdl = compute_average_length(removed_stopwords_docs)

    # query_list = read_file(generate_data_path("query-text"))

    # results = []
    # for query in query_list:
    #     query = remove_stop_words(query.lower())
    #     print(query)
    #     rank = rank_BM25(query, removed_stopwords_docs, N, avgdl)
    #     top_index = []
    #     for i, rsv in rank[:TOP_K]:
    #         print("doc ", i + 1, ":", rsv)
    #         print(docs[i])
    #         top_index.append(i+1)
    #     results.append(top_index)
    #     print("--------------------------------------------------")
    # write_file(generate_data_path("result"), results)
