import pickle
import torch
import numpy as np
import sys
from typing import TypedDict

from .model import EEG_CNN_LSTM_HPO
from .scaler import EEGScaler
from ..config import SCALER_PATH, MODEL_PATH
from ..core.logging_config import get_ml_logger

logger = get_ml_logger(__name__)


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
                'cnn_kernel_size_2': 3,  # checkpoint uses kernel_size 3
                'cnn_kernels_1': 32,  # checkpoint uses 32 kernels
                'cnn_kernels_2': 64,  # checkpoint uses 64 kernels
                'learning_rate': np.float64(0.0017576118123159641),
                'lstm_dense': 128,  # checkpoint uses 128
                'lstm_hidden_size': 128,
                'lstm_layers': 1,  # checkpoint uses 1 layer
                'optimizer': 'rmsprop'
                }
        
    def load_scaler(self):
            logger.info(f"Loading scaler from {SCALER_PATH}")
            try:
                # Ensure the correct module is in sys.modules for unpickling
                import app.ml.scaler
                sys.modules['__main__'].EEGScaler = EEGScaler
                
                with open(SCALER_PATH, 'rb') as file:
                    self.scaler = pickle.load(file)
                logger.info("Scaler loaded successfully")
                return self.scaler
            except FileNotFoundError:
                logger.error(f"Scaler file not found: {SCALER_PATH}")
                raise
            except Exception as e:
                logger.error(f"Failed to load scaler: {e}", exc_info=True)
                raise
        
    def load_model(self, params: Hyperparameters = None):
        logger.info(f"Loading model from {MODEL_PATH}")
        try:
            if params is not None:
                logger.debug("Using custom hyperparameters")
                self.params = params
            else:
                logger.debug("Using default hyperparameters")

            logger.debug(f"Model architecture: cnn_kernels=[{self.params['cnn_kernels_1']},{self.params['cnn_kernels_2']}], "
                        f"lstm_hidden={self.params['lstm_hidden_size']}, lstm_layers={self.params['lstm_layers']}")

            self.model = EEG_CNN_LSTM_HPO(
                cnn_kernels_1=self.params['cnn_kernels_1'],
                cnn_kernel_size_1=self.params['cnn_kernel_size_1'],
                cnn_kernels_2=self.params['cnn_kernels_2'],
                cnn_kernel_size_2=self.params['cnn_kernel_size_2'],
                cnn_dropout=float(self.params['cnn_dropout']),
                cnn_dense=self.params['cnn_dense'],
                lstm_hidden_size=self.params['lstm_hidden_size'],
                lstm_layers=self.params['lstm_layers'],
                lstm_dense=self.params['lstm_dense'],
                dropout=float(self.params['cnn_dropout']),
                num_classes=2,
                )
            device = torch.device("cpu")
            logger.debug(f"Loading model weights on device: {device}")
            weights = torch.load(MODEL_PATH, weights_only=True, map_location=device)
            self.model.load_state_dict(weights)
            self.model.eval()
            logger.info("Model loaded successfully and set to evaluation mode")

            return self.model
        except FileNotFoundError:
            logger.error(f"Model file not found: {MODEL_PATH}")
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            raise
    
    def initialize(self):
        logger.info("Initializing ModelLoader: loading scaler and model")
        try:
            self.load_scaler()
            self.load_model()

            if self.model:
                total_params = sum(p.numel() for p in self.model.parameters())
                logger.info(f"Model loaded with {total_params:,} parameters")
            
            logger.info("ModelLoader initialization complete")
        except Exception as e:
            logger.error(f"ModelLoader initialization failed: {e}", exc_info=True)
            raise
                