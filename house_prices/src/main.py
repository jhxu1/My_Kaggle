from numpy.lib.function_base import average
import pandas as pd
import os
from data_process import DataProcessor
from model import Model, rmsle_cv, AveragingModels
import pdb
import argparse

def load_data():
    # load data
    core_dir = os.path.dirname(__file__)
    root = os.path.dirname(core_dir)
    data_dir = os.path.join(root, "data")
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"), index_col=["Id"])
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"), index_col=["Id"])
    return train_df, test_df

def get_save_path():
    core_dir = os.path.dirname(__file__)
    root = os.path.dirname(core_dir)
    data_dir = os.path.join(root, "data")
    save_path = os.path.join(data_dir, "submission.csv")
    return save_path

def main():
    print("Start main process......")
    train_df, test_df = load_data()
    train_df = DataProcessor("train").pipeline(train_df)
    test_df = DataProcessor("test").pipeline(test_df)

    
    x_train, y_train = train_df.drop(columns=["SalePrice"]), train_df["SalePrice"]

    # model = Model("lgb")
    # model.build(x_train, y_train)
    # single_model_name = ["lasso", "enet", "krr", "gboost", "lgb"]
    single_model_name = ["lasso", "enet", "krr"]
    model_list = [Model(name).model for name in single_model_name]
    model = AveragingModels(model_list)
    model.fit(x_train, y_train)

    x_test = test_df
    y_test = model.predict(x_test)

    assert test_df.shape[0] == 1459, "test df rows {} != 1459".format(test_df.shape[0])

    # generate submisson csv
    save_pd = x_test
    save_pd["SalePrice"] = y_test
    save_pd = save_pd.loc[:, ["SalePrice"]]
    save_path = get_save_path()
    save_pd.to_csv(save_path, index=True)
    print("Finish main process......")


def model_choice():
    train_df, test_df = load_data()
    train_df = DataProcessor("train").pipeline(train_df)
    test_df = DataProcessor("test").pipeline(test_df)
    x_train, y_train = train_df.drop(columns=["SalePrice"]), train_df["SalePrice"]

    # single_model_name = ["lasso", "enet", "krr", "gboost", "lgb"]
    single_model_name = ["lasso", "enet", "krr", "avg"]
    # single_model_name = ["avg"]
    for model_name in single_model_name:
        model = Model(model_name)
        score = rmsle_cv(model.model, x_train, y_train)
        print("\{} score: {:.4f} ({:.4f})\n".format(model_name, score.mean(), score.std()))
    
    x_train, y_train = train_df.drop(columns=["SalePrice"]), train_df["SalePrice"]

def config():
    parser = argparse.ArgumentParser("house price")
    parser.add_argument("--mode", type=str, choices=["model_choice", "predict"], default="predict")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = config()
    mode = args.mode
    if mode == "predict":
        main()
    elif mode == "model_choice":
        model_choice()
    else:
        raise Exception("Unknow mode {}".format(mode))