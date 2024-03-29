from matplotlib.pyplot import axis
import pandas as pd
import os
import pdb
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


class DataProcessor():
    def __init__(self, mode) -> None:
        self.mode = mode

    def pipeline(self, df):
        df = self._feature_screen(df)
        df = self._process_nan(df)
        return df

    def _normalize(self, df):
        pass

    def _feature_screen(self, df):
        """根据分析做特征筛选"""
        print("\033[1;32m [Info]\033[0m Start feature screen, feature_num: {}, num: {}".format(df.columns.size, df.shape[0]))

        # 构造季度销售量
        df["SeSold"] = df["MoSold"].apply(lambda x: (x-1) // 3)

        # 删除部分特征
        # del_feature = ["3SsnPorch", "BsmtHalfBath", "MasVnrArea", "MiscFeature", "MoSold", "PoolArea", "PoolQC", "Utilities", "YrSold", "MiscVal"]
        del_feature = ["MiscFeature"]
        df = df.drop(del_feature, axis=1)

        # 删除脏数据
        if self.mode == "train":
            df = df.dropna(thresh=int(df.columns.size * 0.9))
            # df = df.dropna(subset=['SalePrice'])
            df = df.drop_duplicates()
            df.drop(df[df["GrLivArea"] > 4000].index)
            df.drop(df[df["LotFrontage"] > 200].index)
            df.drop(df[df["LotArea"] > 50000].index)
            df.drop(df[df["BsmtFinSF2"] > 1250].index)
            df.drop(df[df["GarageArea"] > 1200].index)
            df.drop(df[df["OpenPorchSF"] > 400].index)
            df.drop(df[df["EnclosedPorch"] > 400].index)
            df["SalePrice"] = np.log(df["SalePrice"])

        print("\033[1;32m [Info]\033[0m Finish feature screen, feature_num: {}, num: {}".format(df.columns.size, df.shape[0]))
        return df

    @staticmethod
    def _process_nan(df):
        """处理缺失值
        
        - 离散特征使用None填充，后使用label-encoder处理
        - 连续特征使用中位数填充
        """
        print("\033[1;32m [Info]\033[0m Start process nan, feature_num: {}, num: {}".format(df.columns.size, df.shape[0]))
        def if_discrete(column):
            unique_num = len(df[column].unique())
            if unique_num < 20:
                return True
            else:
                if df[column].dtype == 'O':
                    return True
                else:
                    return False

        columns = df.columns
        for column in columns:
            if column == "SalePrice":
                continue
            if if_discrete(column):
                # 离散
                df[column] = df[column].fillna("None")
                lbl = LabelEncoder()
                # lbl = OneHotEncoder()
                lbl.fit(list(df[column].values))
                df[column] = lbl.transform(list(df[column].values))
            else:
                # 连续
                df[column] = df[column].fillna(df[column].median())
                # normalize
                df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())

        print("\033[1;32m [Info]\033[0m Finish process nan, feature_num: {}, num: {}".format(df.columns.size, df.shape[0]))
        return df

def load_data():
    # load data
    core_dir = os.path.dirname(__file__)
    root = os.path.dirname(core_dir)
    data_dir = os.path.join(root, "data")
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"), index_col=["Id"])
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"), index_col=["Id"])
    return train_df, test_df

def test():
    train_df, test_df = load_data()
    processor = DataProcessor()
    train_df = processor.pipeline(train_df)

if __name__ == "__main__":
    test()