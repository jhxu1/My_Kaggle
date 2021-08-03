from numpy.lib.npyio import save
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

FILE_PATH = os.path.dirname(__file__)
SAVE_ROOT = os.path.dirname(FILE_PATH)

def draw(df):
    draw_ratio(df)
    draw_pclass(df)
    draw_sex(df)
    draw_age(df)
    draw_family(df)
    draw_ticket(df)
    draw_cabin(df)
    draw_embarked(df)

def draw_pclass(df):
    fig1, ax = plt.subplots()
    # 经济水平和生还率的关系
    counts = df.groupby(["Pclass", "Survived"])["Sex"].count()
    width = 0.35
    labels = ["upper", "middle", "lower"]
    survive_data = counts.loc[(slice(None), 1)].values
    not_survive_data = counts.loc[(slice(None), 0)].values
    x = x = np.arange(len(labels))  # the label locations
    rects1 = ax.bar(x - width/2, survive_data, width, label='survived')
    rects2 = ax.bar(x + width/2, not_survive_data, width, label='not survived')
    ax.set_ylabel('Counts')
    ax.set_title('Counts by pclass and survive')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig1.savefig(os.path.join(SAVE_ROOT, "./analyze_img/02_survive_pclass.png"))

def draw_sex(df):
    # 性别和生还率的关系
    fig1, ax = plt.subplots()
    counts = df.groupby(["Sex", "Survived"])["Sex"].count()
    # plot
    width = 0.35
    labels = ["female", "male"]
    survive_data = counts.loc[(slice(None), 1)].values
    not_survive_data = counts.loc[(slice(None), 0)].values
    x = x = np.arange(len(labels))  # the label locations
    rects1 = ax.bar(x - width/2, survive_data, width, label='survived')
    rects2 = ax.bar(x + width/2, not_survive_data, width, label='not survived')
    ax.set_ylabel('Counts')
    ax.set_title('Counts by pclass and survive')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig1.savefig(os.path.join(SAVE_ROOT, "./analyze_img/03_survive_sex.png"))

def draw_age(df):
    # 年龄和生还率的关系
    fig1, ax = plt.subplots()
    tmp_df = df[["Age", "Survived"]]
    groupby_df = tmp_df.groupby("Survived")
    save_df = groupby_df.min()
    save_df = save_df.rename(columns={"Age": "Min Age"})
    save_df["Max Age"] = groupby_df.max()["Age"]
    save_df["Mean Age"] = groupby_df.mean()["Age"]
    save_df = save_df.round(2)
    print("==============Analyze Age==============")
    print(save_df)
    save_path = os.path.join(SAVE_ROOT, "./analyze_img/04_survive_age.csv")
    save_df.to_csv(save_path,sep=',')

def draw_family(df):
    tmp_df = df[["Survived", "SibSp", "Parch"]]
    groupby_df = tmp_df.groupby("Survived")
    save_df = groupby_df.mean().round(2)
    save_path = os.path.join(SAVE_ROOT, "./analyze_img/05_survive_family.csv")
    save_df.to_csv(save_path,sep=',')
    print("==============Analyze Family==============")
    print(save_df)

def draw_ticket(df):
    tmp_df = df[["Survived", "Fare"]]
    groupby_df = tmp_df.groupby("Survived")
    save_df = groupby_df.mean().round(2)
    save_path = os.path.join(SAVE_ROOT, "./analyze_img/06_survive_ticket.csv")
    save_df.to_csv(save_path,sep=',')
    print("==============Analyze Ticket==============")
    print(save_df)

def draw_cabin(df):
    tmp_df = df
    count_df = tmp_df.groupby(["Cabin", "Survived"])["Cabin"].count().unstack()
    count_df = count_df.fillna(0)
    count_df["Ratio"]  = count_df[1] / (count_df[1] + count_df[0])
    count_df = count_df.round(2)
    save_path = os.path.join(SAVE_ROOT, "./analyze_img/07_survive_cabin.csv")
    count_df.to_csv(save_path,sep=',')
    print("==============Analyze Cabin==============")
    print(count_df)

    # draw
    fig1, ax = plt.subplots()
    # 经济水平和生还率的关系
    width = 0.5
    labels = count_df.index
    ratio_data = count_df["Ratio"].values
    x = np.arange(len(labels))  # the label locations
    rects2 = ax.bar(x, ratio_data, width, alpha=0.5, color='yellow', edgecolor='red', lw=3)
    ax.set_ylabel('Ratio')
    ax.set_xlabel('Cabin')
    ax.set_title('Survive rate vs Cabin')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 0.9)

    for a, b in zip(x, ratio_data):
        plt.text(x=a, y=b, s="{:.2f}".format(b), ha='center', va='bottom', fontsize=10)

    fig1.savefig(os.path.join(SAVE_ROOT, "./analyze_img/07_survive_cabin.png"))


def draw_embarked(df):
    fig1, ax = plt.subplots()
    counts = df.groupby(["Embarked", "Survived"])["Embarked"].count()
    # plot
    width = 0.35
    labels = counts.unstack().index.to_list()
    survive_data = counts.loc[(slice(None), 1)].values
    not_survive_data = counts.loc[(slice(None), 0)].values
    x = x = np.arange(len(labels))  # the label locations
    rects1 = ax.bar(x - width/2, survive_data, width, label='survived')
    rects2 = ax.bar(x + width/2, not_survive_data, width, label='not survived')
    ax.set_ylabel('Counts')
    ax.set_title('Counts by pclass and survive')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig1.savefig(os.path.join(SAVE_ROOT, "./analyze_img/08_survive_embarked.png"))

def draw_ratio(df, col="Survived"):
    """Survive的比例图"""
    # config
    survive_map = {"0": "Not Survive", "1": "Survive"}
    # 1. 分出是否生还的数量
    data = df[col].value_counts().to_dict()
    labels = data.keys()
    labels = [survive_map[str(k)] for k in data]
    sizes = data.values()
    explode = [0 for _ in range(len(data))]

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax1.set_title("Survived Ratio")
    fig1.savefig(os.path.join(SAVE_ROOT, "./analyze_img/01_survive_ratio.png"))