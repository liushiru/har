import torch
import config
from model import MLP
from preprocess import FeatureDataset
from sklearn.inspection import permutation_importance


model = MLP()
model.load_state_dict(torch.load(config.mlp_model_path))


