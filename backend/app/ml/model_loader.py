import pickle
import torch
import numpy as np
from typing import TypedDict

from .model import EEG_CNN_LSTM_HPO
from .scaler import EEGScaler
from ..config import SCALER_PATH, MODEL_PATH


class Hyperparameters(TypedDict):
    batch_size: int
    cnn_dense: int
    cnn_dropout: np.float64
    cnn_kernel_size_1: int
    cnn_kernel_size_2: int
    cnn_kernels_1: int
    cnn_kernels_2: int
    learning_rate: np.float64
    lstm_dense: int
    lstm_hidden_size: int
    lstm_layers: int
    optimizer: str

class ModelLoader:
    def __init__(self):
        self.scaler: EEGScaler | None = None
        self.model: EEG_CNN_LSTM_HPO | None = None
        self.params: Hyperparameters = self._get_default_params()

    @staticmethod
    def _get_default_params() -> Hyperparameters:
            return {
                'batch_size': 80,
                'cnn_dense': 256,
                'cnn_dropout': np.float64(0.38218620920862145),
                'cnn_kernel_size_1': 5,
                'cnn_kernel_size_2': 5,
                'cnn_kernels_1': 48,
                'cnn_kernels_2': 32,
                'learning_rate': np.float64(0.0017576118123159641),
                'lstm_dense': 32,
                'lstm_hidden_size': 128,
                'lstm_layers': 3,
                'optimizer': 'rmsprop'
                }
        
    def load_scaler(self):
            with open(SCALER_PATH, 'rb') as file:
                self.scaler = pickle.load(file)
            return self.scaler
        
    def load_model(self, params: Hyperparameters = None):
        if params is not None:
            self.params = params

        self.model = EEG_CNN_LSTM_HPO(
            cnn_kernels_1=self.params['cnn_kernels_1'],
            cnn_kernel_size_1=self.params['cnn_kernel_size_1'],
            cnn_kernels_2=self.params['cnn_kernels_2'],
            cnn_dropout=float(self.params['cnn_dropout']),
            cnn_dense=self.params['cnn_dense'],
            lstm_hidden_size=self.params['lstm_hidden_size'],
            lstm_layers=self.params['lstm_layers'],
            lstm_dense=self.params['lstm_dense'],
            dropout=float(self.params['cnn_dropout']),
            num_classes=2,
            )
        device = torch.device("cpu")
        weights = torch.load(MODEL_PATH, weights_only=True, map_location=device)
        self.model.load_state_dict(weights)
        self.model.eval()

        return self.model
    
    def initialize(self):
        self.load_scaler()
        self.load_model()

        if self.model:
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"Model loaded with {total_params:,} parameters")
                