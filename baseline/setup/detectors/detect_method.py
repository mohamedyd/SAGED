from enum import Enum
import os

EXP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..", "experiments"))
DATA_PATH = os.path.abspath(os.path.join(EXP_PATH, "data"))


class DetectMethod(Enum):
    OUTLIER_DETECTOR_IF = 'IF'
    OUTLIER_DETECTOR_SD = 'SD'
    OUTLIER_DETECTOR_IQR = 'IQR'
    MV_DETECTOR = 'mvdetector'
    FAHES_DETECTOR = 'fahes'
    KATARA = 'katara'
    NADEEF = 'nadeef'
    HOLOCLEAN = 'holoclean'
    RAHA = "raha"
    ED2_DETECTOR = 'ed2'
    DBOOST = 'dBoost'
    MIN_K = 'min_k'
    SAGED = 'saged'

    def __str__(self):
        return self.value
