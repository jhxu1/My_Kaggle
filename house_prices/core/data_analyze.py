"""
data analyze and draw
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pdb

class Analyzer():
    core_dir = os.path.dirname(__file__)
    root = os.path.dirname(core_dir)
    data_dir = os.path.join(root, "data")

    def __init__(self) -> None:
        pass

    @classmethod
    def load_df(cls):
        train_df = pd.read_csv(os.path.join(cls.data_dir, "train.csv"), index_col=["Id"])
        test_df = pd.read_csv(os.path.join(cls.data_dir, "test.csv"), index_col=["Id"])
        return train_df, test_df

    @staticmethod
    def print_df_info(df):
        # shape
        print("\033[1;33m [===========shape===========]\033[0m \n{}".format(df.shape))

        # head
        print("\033[1;33m [===========head===========]\033[0m")
        print(df.head())

        # col
        print("\033[1;33m [===========Columns===========]\033[0m")
        print(df.columns)

    @staticmethod
    def analyze_col(series):
        # unique
        unique_num = len(series.unique())
        if unique_num < 100:
            print(series.unique())


if __name__ == "__main__":
    train_df, test_df = Analyzer.load_df()
    pdb.set_trace()
    Analyzer.print_df_info(train_df)