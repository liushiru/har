import os
import torch
epochs = 120
batch_size = 64
learning_rate = 0.0001
val_split = 0.2

num_neighbors = 12
mlp_model_path = os.path.join('Models', 'mlp.h5')
cnn_model_path = os.path.join('Models', 'cnn.h5')
model_path = os.path.join('Models', 'mlp.h5')





# raw_data_path = os.path.join('Data', 'RawExtract', 'windowed_raw.csv')
# raw_data_path = os.path.join('Data', 'RawExtract', 'raw_25hz.csv')
raw_data_path = os.path.join('Data', 'RawExtract', 'raw_25hz_id.csv')

# input_size = (6, 128)
input_size = (6,64)


num_label_cols = 2

num_users = 5

confusion_matrix_path = os.path.join('Data', 'ConfusionMatrix', 'matrix.csv')

