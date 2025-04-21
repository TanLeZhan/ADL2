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
    "DATA_2": {
        "synthesizer": "TVAE",
        "epochs": 1000,
        "n_synthetic": 10000,
        "od_majority": IQRDetector(factor=1.5),
        "od_minority": IQRDetector(factor=1.5),
        "scaler": "Robust"
    },
    "DATA_3": {
        "synthesizer": "TVAE",
        "epochs": 2500,
        "n_synthetic": 10000,
        "od_majority": IQRDetector(factor=1.5),
        "od_minority": IQRDetector(factor=1.5),
        "scaler": "Robust"
    },
    "DATA_4": {
        "synthesizer": "TVAE",
        "epochs": 1000,
        "n_synthetic": 10000,
        "od_majority": IQRDetector(factor=1.5),
        "od_minority": IQRDetector(factor=1.5),
        "scaler": "Robust"
    },
    "DATA_5": {
        "synthesizer": "CTGAN",
        "epochs": 1000,
        "n_synthetic": 10000,
        "od_majority": IQRDetector(factor=1.5),
        "od_minority": IQRDetector(factor=1.5),
        "scaler": "Robust"
    },
    "DATA_6": {
        "synthesizer": "CTGAN",
        "epochs": 2500,
        "n_synthetic": 10000,
        "od_majority": IQRDetector(factor=1.5),
        "od_minority": IQRDetector(factor=1.5),
        "scaler": "Robust"
    },
    "DATA_7": {
        "synthesizer": "CTGAN",
        "epochs": 2500,
        "n_synthetic": 10000,
        "od_majority": IQRDetector(factor=1),
        "od_minority": IQRDetector(factor=1),
        "scaler": "Robust"
    },
    "DATA_8": {
        "synthesizer": "CTGAN",
        "epochs": 1000,
        "n_synthetic": 10000,
        "od_majority": IQRDetector(factor=1),
        "od_minority": IQRDetector(factor=1),
        "scaler": "Robust"
    }
}