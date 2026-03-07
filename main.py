import numpy as np

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

class Word2VecSGNS:
    def __init__(self, vocab_size, embedding_dim, learning_rate=0.01):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lr = learning_rate
        self.W1 = np.random.uniform(-0.5/embedding_dim, 0.5/embedding_dim, (vocab_size, embedding_dim))
        self.W2 = np.zeros((vocab_size, embedding_dim))

    def train_step(self, center_word_idx, pos_context_idx, neg_context_indices):
        v_c = self.W1[center_word_idx]
        context_indices = [pos_context_idx] + neg_context_indices
        u_contexts = self.W2[context_indices]
        
        labels = np.zeros(len(context_indices))
        labels[0] = 1.0 
        
        dot_products = np.dot(u_contexts, v_c)
        predictions = sigmoid(dot_products)
        
        errors = predictions - labels
        
        grad_W2 = np.outer(errors, v_c)
        grad_W1 = np.dot(errors, u_contexts)
        
        self.W1[center_word_idx] -= self.lr * grad_W1
        
        for i, ctx_idx in enumerate(context_indices):
            self.W2[ctx_idx] -= self.lr * grad_W2[i]
