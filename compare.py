from collections import Counter
from time import time
import scipy.sparse as sp


def read_data(path):
    with open(path, "r") as f:
        return [line.lower().split() for line in f.readlines()]


def index_data(data):
    idx2word = list(set([word for line in data for word in line]))
    word2idx = {word: idx for idx, word in enumerate(idx2word)}

    return (idx2word, word2idx)


def count_data(index, data):
    x = []
    y = []
    vals = []
    for doc_id, doc in enumerate(data):
        counts = Counter(doc)
        for word, count in counts.items():
            x.append(index[word])
            y.append(doc_id)
            vals.append(count)
    return sp.csr_matrix((vals, (x, y)))


t0 = time()
data = read_data("MovieSummaries/plot_summaries.txt")
print(f"Reading took {time() - t0}")


t0 = time()
idx2word, word2idx = index_data(data)
print(f"Indexing took {time() - t0}")

t0 = time()
count_mat = count_data(word2idx, data)
print(f"Counting took {time() - t0}")

print(count_mat.shape)
