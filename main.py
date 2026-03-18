import numpy as np
import urllib.request
import re
from collections import Counter
import random


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


class Word2VecSGNS:
    def __init__(self, vocab_size, embedding_dim, learning_rate=0.01):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lr = learning_rate
        self.W1 = np.random.uniform(-0.5 / embedding_dim, 0.5 / embedding_dim, (vocab_size, embedding_dim))
        self.W2 = np.zeros((vocab_size, embedding_dim))

    def train_step(self, center_word_idx, pos_context_idx, neg_context_indices):
        v_c = self.W1[center_word_idx]   # representation of the center word
        context_indices = [pos_context_idx] + neg_context_indices
        u_contexts = self.W2[context_indices]  # representation of the context words (from a different matrix)

        labels = np.zeros(len(context_indices))
        labels[0] = 1.0

        dot_products = np.dot(u_contexts, v_c)  # forward pass
        predictions = sigmoid(dot_products)
        errors = predictions - labels

        grad_W2 = np.outer(errors, v_c)  # derivatives for context words
        grad_W1 = np.dot(errors, u_contexts)  # derivatives for center word

        self.W1[center_word_idx] -= self.lr * grad_W1
        for i, ctx_idx in enumerate(context_indices):
            self.W2[ctx_idx] -= self.lr * grad_W2[i]

url = "https://www.gutenberg.org/files/11/11-0.txt"
response = urllib.request.urlopen(url)
raw_text = response.read().decode('utf-8')
text_words = re.sub(r'[^a-zA-Z\s]', '', raw_text).lower().split()

vocab_size = 1000
word_counts = Counter(text_words)
top_words = [w for w, count in word_counts.most_common(vocab_size - 1)]
top_words.append('UNKNOWN')   # 999 most frequent words and unknown for all the others

word2idx = {w: i for i, w in enumerate(top_words)}
idx2word = {i: w for i, w in enumerate(top_words)}

data = [word2idx.get(w, word2idx['UNKNOWN']) for w in text_words]

embedding_dim = 30
learning_rate = 0.05
epochs = 3
window_size = 2
num_neg_samples = 4

model = Word2VecSGNS(vocab_size, embedding_dim, learning_rate)

for epoch in range(epochs):
    for i, center_word_idx in enumerate(data):  # now we go through each pair of (center word, positive context word and update the weights
        start = max(0, i - window_size)
        end = min(len(data), i + window_size + 1)
        for j in range(start, end):
            if i == j: continue
            pos_context_idx = data[j]

            neg_context_indices = []
            while len(neg_context_indices) < num_neg_samples:
                rand_idx = random.randint(0, vocab_size - 1)
                if rand_idx != pos_context_idx and rand_idx != center_word_idx:
                    neg_context_indices.append(rand_idx)

            model.train_step(center_word_idx, pos_context_idx, neg_context_indices)

def get_similar_words(word, model, word2idx, idx2word, top_k=5):
    if word not in word2idx:
        return f"'{word}' not in vocabulary."

    word_idx = word2idx[word]
    word_vec = model.W1[word_idx]

    similarities = {}
    for i in range(model.vocab_size):
        if i == word_idx: continue
        other_vec = model.W1[i]

        dot_product = np.dot(word_vec, other_vec)
        norm_a = np.linalg.norm(word_vec)
        norm_b = np.linalg.norm(other_vec)
        sim = dot_product / (norm_a * norm_b)

        similarities[idx2word[i]] = sim

    sorted_sims = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return sorted_sims[:top_k]


test_words = ['alice', 'queen', 'rabbit']
for test_word in test_words:
    print(f"\n5 most similar words to '{test_word}':")
    results = get_similar_words(test_word, model, word2idx, idx2word)
    for sim_word, score in results:
        print(f"  {sim_word} ({score:.3f})")
