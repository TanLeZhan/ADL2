import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Preprocessing_Functions2 import IQRDetector

STRATEGIES = {
    "DATA_1": {
        "synthesizer": "TVAE",
        "epochs": 1000,
        "n_synthetic": 10000,
        "od_majority": IQRDetector(factor=1),
        "od_minority": IQRDetector(factor=1),
        "scaler": "Robust"
    },
    "DATA_1_2": {
        "synthesizer": "TVAE",
        "epochs": 1000,
        "n_synthetic": 10000,
        "od_majority": IQRDetector(factor=1.5),
        "od_minority": IQRDetector(factor=1.5),
        "scaler": "Standard"
    },
    "DATA_1_3": {
        "synthesizer": "TVAE",
        "epochs": 2500,
        "n_synthetic": 10000,
        "od_majority": IQRDetector(factor=1.5),
        "od_minority": IQRDetector(factor=1.5),
        "scaler": "Standard"
    },
    "DATA_1_4": {
        "synthesizer": "TVAE",
        "epochs": 1000,
        "n_synthetic": 10000,
        "od_majority": IQRDetector(factor=1.5),
        "od_minority": IQRDetector(factor=1.5),
        "scaler": "Standard"
    },
    "DATA_1_5": {
        "synthesizer": "CTGAN",
        "epochs": 1000,
        "n_synthetic": 10000,
        "od_majority": IQRDetector(factor=1.5),
        "od_minority": IQRDetector(factor=1.5),
        "scaler": "Standard"
    },
    "DATA_1_6": {
        "synthesizer": "CTGAN",
        "epochs": 2500,
        "n_synthetic": 10000,
        "od_majority": IQRDetector(factor=1.5),
        "od_minority": IQRDetector(factor=1.5),
        "scaler": "Standard"
    },
    "DATA_1_7": {
        "synthesizer": "CTGAN",
        "epochs": 2500,
        "n_synthetic": 10000,
        "od_majority": IQRDetector(factor=1),
        "od_minority": IQRDetector(factor=1),
        "scaler": "Robust"
    },
    "DATA_1_8": {
        "synthesizer": "CTGAN",
        "epochs": 1000,
        "n_synthetic": 10000,
        "od_majority": IQRDetector(factor=1),
        "od_minority": IQRDetector(factor=1),
        "scaler": "Robust"
    },
    "DATA_9": {
        "synthesizer": "TVAE",
        "epochs": 1000,
        "n_synthetic": 10000,
        "od_majority": IQRDetector(factor=1.2),
        "od_minority": IQRDetector(factor=1.8),
        "scaler": "Standard"
    },
    "DATA_10": {
        "synthesizer": "CTGAN",
        "epochs": 1000,
        "n_synthetic": 10000,
        "od_majority": IQRDetector(factor=1),
        "od_minority": IQRDetector(factor=1.5),
        "scaler": "Standard"
    }
}