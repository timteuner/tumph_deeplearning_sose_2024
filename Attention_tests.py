import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import grad
import numpy as np

#HYPERPARAMETERS
embed_dim = 4
attention_dim = 6

#DATA GENERATION

def generate_data(N):
    X = np.random.randint(0, 10, size=(N, 10))
    counts_2 = np.sum(X == 2, axis=1)
    counts_4 = np.sum(X == 4, axis=1)
    y = (counts_2 > counts_4).astype(int).reshape(-1, 1)
    return X, y

test_X, test_y = generate_data(12)
print("Shape of test_X:", test_X.shape)
print("Shape of test_y:", test_y.shape)

#EMBEDDING LAYER

embed = torch.nn.Embedding(10, embed_dim)
test_vector = generate_data(12)[0]
test_embedded = embed(torch.tensor(test_vector, dtype=torch.long))
print("Shape of test_embedded:", test_embedded.shape)

#ATTENTION MECHANISM

key_layer = nn.Linear(embed_dim, attention_dim)
value_layer = nn.Linear(embed_dim, attention_dim)

single_query = query = torch.randn(1, attention_dim)

def attention(query, keys, values):
    scores = torch.matmul(query, keys.transpose(-2, -1)) / np.sqrt(attention_dim)
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, values)
    

test_output = attention(single_query, key_layer(test_embedded), value_layer(test_embedded))
print("Shape of test_output:", test_output.shape)

#verif_output = attention(query, verif_keys, verif_values)
#print("Verification Output Shape:", verif_output.shape)
