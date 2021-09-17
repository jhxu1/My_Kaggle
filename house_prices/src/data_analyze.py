"""
data analyze and draw
分析结论：
删除特征为：["3SsnPorch", "BsmtHalfBath", "MasVnrArea", "MiscFeature", "MoSold", "PoolArea", "PoolQC", "Utilties", "YrSold"]
转换出特征：季度售卖量["SeSold"]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pdb
import math

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


class Drawer():
    def __init__(self, df, save_dir) -> None:
        self.df = df
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.preprocess()

    def preprocess(self):
        self._df_preprocess()
        self._feature_classify()

    def _feature_classify(self):
        """特征归类，离散特征和连续特征"""
        df = self.df
        columns = df.columns
        continue_feature, discrete_feature = [], []
        for column in columns:
            if column == "SalePrice":
                continue
            unique_num = len(df[column].unique())
            if unique_num < 20:
                discrete_feature.append(column)
            else:
                if df[column].dtype == 'O':
                    discrete_feature.append(column)
                else:
                    continue_feature.append(column)
        print("Continue feature num: {}, discrete feature num: {}".format(len(continue_feature), len(discrete_feature)))
        self.con_feat, self.dis_feat = continue_feature, discrete_feature


    def simple_draw(self):
        col_num = 6
        row_num = math.ceil(len(self.con_feat) / col_num) 
        fig, axs = plt.subplots(row_num, col_num, figsize=(36, 24)) 
        for idx, column in enumerate(self.con_feat):
            row = idx // col_num
            col = idx - row * col_num
            try:
                self.draw_continue_feature(column, axs[row][col])
            except Exception as e:
                print(e)
                print("\033[1;31m [Error]\033[0m wrong continue column {}".format(column))
        plt.subplots_adjust(wspace=0.3, hspace=0.3)#调整子图间距
        fig.savefig(os.path.join(self.save_dir, "continue.png"))
        plt.close(fig)

        row_num = math.ceil(len(self.dis_feat) / col_num)
        fig, axs = plt.subplots(row_num, col_num, figsize=(36, 24)) 
        for idx, column in enumerate(self.dis_feat):
            row = idx // col_num
            col = idx - row * col_num
            try:
                self.draw_discrete_feature(column, axs[row][col])
            except Exception as e:
                print(e)
                print("\033[1;31m [Error]\033[0m wrong discrete column {}".format(column))
        plt.subplots_adjust(wspace=0.3, hspace=0.6)#调整子图间距
        fig.savefig(os.path.join(self.save_dir, "discrete.png"))
        plt.close(fig)

    def _df_preprocess(self):
        self.df["SalePrice"] = (self.df["SalePrice"]).round(2)
        # 提出重复样本
        self.df.drop_duplicates()
    
    def draw_discrete_feature(self, column, ax):
        """初步绘制，离散特征"""
        # fig1, ax = plt.subplots()
        tmp_df = self.df[[column, "SalePrice"]]

        # 去除掉出现<3次的特征
        feat_counts = tmp_df[column].value_counts()
        del_feat = feat_counts[feat_counts < 3].index.to_list()
        for feat in del_feat:
            tmp_df = tmp_df[tmp_df[column] != feat]

        sub_class_df = tmp_df.groupby([column])["SalePrice"].agg(["mean", "std"])
        x = np.arange(sub_class_df.shape[0])
        width = 0.6
        p1 = ax.bar(x, sub_class_df["mean"].to_list(), width, yerr=sub_class_df["std"].to_list(), label=column)

        # title = "{} vs. SalePrice".format(column)
        # ax.set_ylabel('SalePrice (Pound)')
        ax.set_title(column)
        ax.set_xticks(x)
        # ax.set_xticklabels(sub_class_df.index.to_list())
        # ax.bar_label(p1, label_type='center')
        # fig1.savefig(os.path.join(self.save_dir, "{}.png".format(title.replace(" ", "_"))))
        # plt.close(fig1)

    def draw_continue_feature(self, column, ax):
        """初步绘制，连续特征"""
        #TODO: 连续特征做分箱处理
        # fig1, ax = plt.subplots()
        sub_df = self.df[[column, "SalePrice"]]
        sub_df = sub_df.sort_values(by=column, ascending=True)

        # p1 = plt.plot(sub_df[column], sub_df["SalePrice"])
        p2 = ax.scatter(sub_df[column], sub_df["SalePrice"], color="red", marker="o", linewidths=0.2)
        # title = '{} vs. SalePrice'.format(column)
        # ax.set_ylabel('SalePrice (Pound)')
        ax.set_title(column)
        # ax.set_xlabel(column)
        # fig1.savefig(os.path.join(self.save_dir, "{}.png".format(title.replace(" ", "_"))))
        # plt.close(fig1)

    def _draw_lot_Area(self):
        pass

def debug_col(df, column_name):
    column = df[column_name]
    print("column: {}".format(column_name))
    print("每个特征出现的次数：\n", column.value_counts())

def seasonal_feature(df):
    def map_func(x):
        return (x-1) // 3
    df["SeSold"] = df["MoSold"].apply(map_func)
    return df

def main():
    src_dir = os.path.dirname(__file__)
    root_dir = os.path.dirname(src_dir)
    save_dir = os.path.join(root_dir, "analyze_data")
    os.makedirs(save_dir, exist_ok=True)

    train_df, test_df = Analyzer.load_df()
    # Analyzer.print_df_info(train_df)

    drawer = Drawer(train_df, save_dir)
    drawer.simple_draw()

def log_draw():
    src_dir = os.path.dirname(__file__)
    root_dir = os.path.dirname(src_dir)
    save_dir = os.path.join(root_dir, "log_analyze_data")
    os.makedirs(save_dir, exist_ok=True)

    train_df, test_df = Analyzer.load_df()

    # simple process

    train_df["SalePrice"] = np.log(train_df["SalePrice"])
    # 删除离群点
    train_df = train_df.drop(train_df[train_df["GrLivArea"] > 4000].index)
    # train_df

    drawer = Drawer(train_df, save_dir)
    drawer.simple_draw()

def test():
    src_dir = os.path.dirname(__file__)
    root_dir = os.path.dirname(src_dir)
    save_dir = os.path.join(root_dir, "analyze_data")
    os.makedirs(save_dir, exist_ok=True)

    train_df, test_df = Analyzer.load_df()
    debug_col(train_df, "MiscFeature")
    debug_col(train_df, "PoolArea")
    debug_col(train_df, "PoolQC")
    
    train_df = seasonal_feature(train_df)
    drawer = Drawer(train_df, save_dir)
    drawer.draw_discrete_feature("SeSold")

if __name__ == "__main__":
    # main()
    # test()
    log_draw()
