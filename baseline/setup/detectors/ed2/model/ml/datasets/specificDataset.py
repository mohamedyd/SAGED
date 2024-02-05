import numpy as np
import pandas as pd
from ml.configuration.Config import Config

from ml.datasets.DataSet import DataSet

class SpecificDataset(DataSet):
    
    

    def __init__(self, name, dirty_df, clean_df):
        clean_df = clean_df
        dirty_df = dirty_df
        name = name

        super(SpecificDataset, self).__init__(name, dirty_df, clean_df)

    def validate(self):
        print("validate")