import pdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from core.data_draw import draw
from core.data_process import process, split
from core.model import Model, evaluate
import os

def main():
    file_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(file_path, "data", "train.csv")
    df = pd.read_csv(data_path)
    df = process(df)
    draw(df)
    print(df.head())

    # train
    x_train, x_test, y_train, y_test = split(df)
    model = Model()
    model.build(x_train, y_train)
    y_pred = model.predict(x_test)
    evaluate(y_pred, y_test)

if __name__ == '__main__':
    main()