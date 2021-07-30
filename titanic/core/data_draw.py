import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def draw(df):
    draw_ratio(df)
    draw_pclass(df)
    draw_sex(df)
    draw_age(df)

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
    fig1.savefig("./analyze_img/survive_pclass.png")

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
    fig1.savefig("./analyze_img/survive_sex.png")

def draw_age(df):
    # 年龄和生还率的关系
    fig1, ax = plt.subplots()
    tmp_df = df[["Age", "Survived"]]
    groupby_df = tmp_df.groupby("Survived")
    print("==========Max age==========\n{}".format(groupby_df.max()))
    print("==========Min age==========\n{}".format(groupby_df.min()))
    print("==========Mean age==========\n{}".format(groupby_df.mean()))

    # 兄弟姐妹数量和生还率的关系

    # 父母、孩子数量和生还率的关系


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
    fig1.savefig("./analyze_img/survive_ratio.png")