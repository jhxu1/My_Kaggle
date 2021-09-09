import pandas as pd
import os
from data_process import DataProcessor
from model import Model

def load_data():
    # load data
    core_dir = os.path.dirname(__file__)
    root = os.path.dirname(core_dir)
    data_dir = os.path.join(root, "data")
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"), index_col=["Id"])
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"), index_col=["Id"])
    return train_df, test_df

def main():
    train_df, test_df = load_data()
    processor = DataProcessor()
    train_df = processor.pipeline(train_df)
    test_df = processor.pipeline(test_df)

    
    x_train, y_train = train_df.drop(columns=["SalePrice"]), train_df["SalePrice"]

    model = Model("lasso")
    model.build(x_train, y_train)
    x_test = test_df
    y_test = model.predict(x_test)
    print(y_train.mean())
    print(y_test.mean())

    for model_name in ["lasso", "enet", "krr"]:
        model = Model(model_name)
        score = model.rmsle_cv(x_train, y_train / 1e6)
        print("\{} score: {:.4f} ({:.4f})\n".format(model_name, score.mean(), score.std()))

if __name__ == "__main__":
    main()