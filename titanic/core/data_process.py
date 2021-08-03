import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def process(df):
    df = _nan_process(df)
    df = _feature_engineer(df)
    return df

def split(df):
    x_train, x_test, y_train, y_test = train_test_split(df.drop(columns=["Survived"]), df["Survived"], 
                                                        test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test

def _nan_process(df):
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

def _feature_engineer(df):
    df["Cabin"] = df["Cabin"].apply(lambda s:s[0])
    df = df.drop(["PassengerId", "Name", "Ticket"], axis=1)
    return df

