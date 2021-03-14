import os

fs = 40
wl = 60
num_dancers = 5
num_moves = 3
num_features=54


epochs = 80
batch_size = 128
learning_rate = 0.0001
val_split = 0.2

num_neighbors = 12
mlp_model_path = os.path.join('Models', 'mlp.h5')
cnn_model_path = os.path.join('Models', 'cnn.h5')
svm_model_path = os.path.join('Models', 'svm.p')


model_name = 'MLP_LD'
model_path = None

if model_name[:3] == 'MLP':
    model_path = mlp_model_path


scalar_path = os.path.join('Data', 'RawExtract', 'scalar.pkl')

# raw_data_path = os.path.join('Data', 'RawExtract', 'windowed_raw.csv')
# raw_data_path = os.path.join('Data', 'RawExtract', 'raw_25hz.csv')
raw_data_path = os.path.join('Data', 'RawExtract', 'raw_25hz_id.csv')

# input_size = (6, 128)
input_size = (6,64)


num_label_cols = 2

K = 4

confusion_matrix_path = os.path.join('Data', 'Analysis', 'matrix.csv')
confusion_matrix_trad_path = os.path.join('Data', 'ConfusionMatrix', 'matrix_trad.csv')

