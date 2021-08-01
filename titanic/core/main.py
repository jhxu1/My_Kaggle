import pdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_draw import draw
from data_process import nan_process
import os

def main():
    file_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(file_path, "..", "data", "train.csv")
    df = pd.read_csv(data_path)
    # draw_ratio(df)

    df = nan_process(df)
    print(df)
    draw(df)

if __name__ == '__main__':
    main()