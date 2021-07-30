import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def nan_process(df):
    """
    不存在NAN的Pclass、Name、Sex、SibSp、Parch、Ticket、Fare
    Age: 714个值，177个NAN，使用平均数替代
    Cabin: 687真值，204NAN，因为是离散值，使用某个固定值替代
    Embarked: 889真值，2NAN，使用众数替代
    """
    df["Age"] = df["Age"].fillna(df["Age"].dropna().mean())
    df["Cabin"] = df["Cabin"].fillna("N")
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].dropna().mode().values[0])
    return df


