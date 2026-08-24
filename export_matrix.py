import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D
import numpy as np

vocab_size = 10000
embedding_dim = 300

print("Ladowanie modelu i wag...")
model = Sequential([
    Embedding(vocab_size, embedding_dim),
    GlobalAveragePooling1D(),
    Dense(vocab_size, activation='softmax')
])
model.build((None, None))
model.load_weights("word2vec.h5")

embedding_matrix = model.layers[0].get_weights()[0]

np.save("embedding_matrix.npy", embedding_matrix)
print("Sukces! Macierz zapisana jako embedding_matrix.npy")