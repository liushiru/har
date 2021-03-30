import os

fs = 20
wl = 30
num_dancers = 2
num_moves = 3
num_features = 84
num_axis = 12


epochs = 100
batch_size = 128
learning_rate = 0.0001
val_split = 0.5

num_neighbors = 12
mlp_model_path = os.path.join('Models', 'mlp.h5')
cnn_model_path = os.path.join('Models', 'cnn.h5')
svm_model_path = os.path.join('Models', 'svm.p')

inference_model_path = os.path.join('Models', 'mlp.h5')


model_name = 'MLP'
model_path = None

if model_name[:3] == 'MLP':
    model_path = mlp_model_path


scalar_path = os.path.join('Data', 'RawExtract', 'scaler.pkl')

# raw_data_path = os.path.join('Data', 'RawExtract', 'windowed_raw.csv')
# raw_data_path = os.path.join('Data', 'RawExtract', 'raw_25hz.csv')
raw_data_path = os.path.join('Data', 'RawExtract', 'raw_25hz_id.csv')

# input_size = (6, 128)
input_size = (6,64)


num_label_cols = 2

K = 4

confusion_matrix_path = os.path.join('Data', 'Analysis', 'matrix.csv')
confusion_matrix_trad_path = os.path.join('Data', 'ConfusionMatrix', 'matrix_trad.csv')

