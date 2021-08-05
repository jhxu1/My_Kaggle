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

    # test
    y_pred = model.predict(x_test)
    evaluate(y_pred, y_test)

    # get result
    result_path = os.path.join(file_path, "data", "test.csv")
    save_path = os.path.join(file_path, "data", "test_with_result.csv")
    df = pd.read_csv(result_path)
    processd_df = process(df)
    y_pred = model.predict(processd_df)
    survived_col = pd.DataFrame(y_pred, columns=["Survived"])
    survived_col.insert(0, 'PassengerId', df["PassengerId"], allow_duplicates=False)
    print(survived_col.head())
    # print(df.head())
    survived_col.to_csv(save_path, index=False)



if __name__ == '__main__':
    main()