import os
import torch
epochs = 200
batch_size = 64
learning_rate = 0.001
val_split = 0.2

num_neighbors = 12
mlp_model_path = os.path.join('Models', 'mlp.h5')
cnn_model_path = os.path.join('Models', 'cnn.h5')
