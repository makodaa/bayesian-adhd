import json
import pickle
import sys
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import torch

from ..config import MODEL_PATH, PARAMETERS_PATH
from ..core.logging_config import get_ml_logger
from .model import EEGCNNLSTM

logger = get_ml_logger(__name__)


class Hyperparameters(TypedDict):
    batch_size: int
    cnn_dense: int
    cnn_dropout: float
    cnn_kernel_size_1: int
    cnn_kernel_size_2: int
    cnn_kernels_1: int
    cnn_kernels_2: int
    learning_rate: float
    lstm_dense: int
    lstm_hidden_size: int
    lstm_layers: int
    optimizer: str
    dropout: float


class ModelLoader:
    def __init__(self):
        self.model: EEGCNNLSTM | None = None
        self.params: Hyperparameters = self._get_default_params()

    @staticmethod
    def _get_default_params() -> Hyperparameters:
        return {
            "batch_size": 48,
            "cnn_dense": 256,
            "cnn_dropout": 0.24205666720642469,
            "cnn_kernel_size_1": 5,
            "cnn_kernel_size_2": 5,
            "cnn_kernels_1": 32,
            "cnn_kernels_2": 64,
            "learning_rate": 0.00013975559179342043,
            "lstm_dense": 64,
            "lstm_hidden_size": 128,
            "lstm_layers": 2,
            "optimizer": "adam",
            "dropout": 0.24205666720642469
        }

    def load_model(self, params: Hyperparameters | None = None):
        logger.info(f"Loading model from {MODEL_PATH}")
        try:
            if params is not None:
                logger.debug("Using custom hyperparameters")
                self.params = params
            else:
                logger.debug("Using default hyperparameters")

            logger.debug(
                f"Model architecture: cnn_kernels=[{self.params['cnn_kernels_1']},{self.params['cnn_kernels_2']}], "
                f"lstm_hidden={self.params['lstm_hidden_size']}, lstm_layers={self.params['lstm_layers']}"
            )

            self.model = EEGCNNLSTM(
                cnn_kernels_1=self.params["cnn_kernels_1"],
                cnn_kernel_size_1=self.params["cnn_kernel_size_1"],
                cnn_kernels_2=self.params["cnn_kernels_2"],
                cnn_kernel_size_2=self.params["cnn_kernel_size_2"],
                cnn_dropout=float(self.params["cnn_dropout"]),
                cnn_dense=self.params["cnn_dense"],
                lstm_hidden_size=self.params["lstm_hidden_size"],
                lstm_layers=self.params["lstm_layers"],
                lstm_dense=self.params["lstm_dense"],
                dropout=float(self.params["cnn_dropout"]),
                num_classes=4,
            )
            # import os
            # print(f"The current working directory: %s" % ('\n'.join([str(x) for x in (Path(os.getcwd())).glob('**/*')]),))
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
        logger.info("Initializing ModelLoader: loading model")
        try:
            with open(PARAMETERS_PATH, "rt") as file:
                content = file.read()
                loaded_parameters = json.loads(content)
                self.load_model(cast(Hyperparameters, loaded_parameters))

            if self.model:
                total_params = sum(p.numel() for p in self.model.parameters())
                logger.info(f"Model loaded with {total_params:,} parameters")

            logger.info("ModelLoader initialization complete")
        except Exception as e:
            logger.error(f"ModelLoader initialization failed: {e}", exc_info=True)
            raise
