[竞赛页面](https://www.kaggle.com/c/titanic)<br />[一个写的比较好的pipeline](https://www.kaggle.com/aimack/simple-ml-pipeline-for-titanic)<br />[知乎上的pipeline，分析比较详细](https://zhuanlan.zhihu.com/p/31743196)
<a name="h6zq4"></a>
## 1. 数据概览
<a name="REdBW"></a>
### 1.1 数据概览
| PassengerId | Survived | Pclass | Name | Sex | Age | SibSp | Parch | Ticket | Fare | Cabin | Embarked |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0 | 3 | Braund, Mr. Owen Harris | male | 22 | 1 | 0 | A/5 21171 | 7.25 |  | S |
| 2 | 1 | 1 | Cumings, Mrs. John Bradley (Florence Briggs Thayer) | female | 38 | 1 | 0 | PC 17599 | 71.2833 | C85 | C |
| 3 | 1 | 3 | Heikkinen, Miss. Laina | female | 26 | 0 | 0 | STON/O2. 3101282 | 7.925 |  | S |
| 4 | 1 | 1 | Futrelle, Mrs. Jacques Heath (Lily May Peel) | female | 35 | 1 | 0 | 113803 | 53.1 | C123 | S |
| 5 | 0 | 3 | Allen, Mr. William Henry | male | 35 | 0 | 0 | 373450 | 8.05 |  | S |
| 6 | 0 | 3 | Moran, Mr. James | male |  | 0 | 0 | 330877 | 8.4583 |  | Q |
| 7 | 0 | 1 | McCarthy, Mr. Timothy J | male | 54 | 0 | 0 | 17463 | 51.8625 | E46 | S |

- 891行
- 12列
<a name="3D3hL"></a>
### 1.2 字段分析

- `PassengerId`：乘客ID，无具体意义
- `Survived`：输出的结果，乘客是否生还
   - `1`：生还
   - `0 `：未生还
- `Pclass`：乘客的经济水平
   - `1`：一等
   - `2`：二等
   - `3` ：三等
- `Name`：乘客名称
- `Sex`：性别
   - `male`：男性
   - `female`：女性
- `Age`：年龄，整数连续数字
- `SibSp`：兄弟姐妹和配偶的总数
- `Parch`：父母和孩子的综述
- `Ticket`：票根
- `Fare`：票价
- `Cabin`：船舱号
- `Embarked`：登船地点，有S/C/Q三个地方。
<a name="sY30f"></a>
## 2. 缺失值补充
一共有三个字段有缺失值

| 字段 | 缺失值比例 | 补全方法 |
| --- | --- | --- |
| Age | 19.19% | 平均数 |
| Cabin | 22.90% | Cabin是李三吱，使用固定值替代 |
| Embarked | 0.02% | 影响较小，使用众数替代 |

<a name="eAH08"></a>
## 3. 数据分析
为了要得到生还率与不同特征之前的关系，我们首先要根据已有的数据，找到这些规律，可以想到的有这些。

1. 首先观察生还率的分布，看下数据是否平衡。
1. 观察生还率与乘客经济水平Pclass的关系。
1. 观察生还率与乘客性别的关系。
1. 观察生还率与年龄的关系。
1. 观察生还率与兄弟姐妹、父母孩子的关系。
1. 观察生还率与票价的关系
1. 观察生还率与船舱号的关系。
1. 观察生还率与登船地点的关系。
<a name="E47m3"></a>
### 3.1 生还率分布
![01_survive_ratio.png](https://cdn.nlark.com/yuque/0/2021/png/21513897/1627803480487-2b0a5d45-3263-4ce6-b1f3-3170afb22f83.png#height=480&id=h7htt&margin=%5Bobject%20Object%5D&name=01_survive_ratio.png&originHeight=480&originWidth=640&originalType=binary&ratio=1&size=23299&status=done&style=none&width=640)<br />**分析**<br />生还率与非生还率比例还算平均，由于题目中就给那么点数据，也不需要调数据维护样本标签均衡。
<a name="P9kVY"></a>
### 3.2 生还与乘客经济水平的关系
![02_survive_pclass.png](https://cdn.nlark.com/yuque/0/2021/png/21513897/1627803520861-3d05d91a-bb82-4cb1-90ee-dfb2314a414e.png#height=480&id=TgPof&margin=%5Bobject%20Object%5D&name=02_survive_pclass.png&originHeight=480&originWidth=640&originalType=binary&ratio=1&size=19019&status=done&style=none&width=640)<br />**分析**

- 地经济水平乘客的人数要多
- 低经济水平的乘客死亡率要明显高一些
<a name="WmyAH"></a>
### 3.3 生还与乘客性别的关系
![03_survive_sex.png](https://cdn.nlark.com/yuque/0/2021/png/21513897/1627803581701-02c06fa5-6a89-4780-9447-3e4511d57915.png#height=480&id=PBePI&margin=%5Bobject%20Object%5D&name=03_survive_sex.png&originHeight=480&originWidth=640&originalType=binary&ratio=1&size=16166&status=done&style=none&width=640)<br />**分析**

- 女性的生还率要明显高于男性
<a name="kJquq"></a>
### 3.4 生还与年龄的关系
| 是否生还 | 最小年龄 | 最大年龄 | 平均年龄 |
| --- | --- | --- | --- |
| 是 | 1.0 | 74.0 | 30.42 |
| 否 | 0.7 | 80.0 | 28.55 |

**分析**

- 年龄分布比较平均
<a name="Ts2zh"></a>
### 3.5 生还与家庭成员人数的关系
| 是否生还 | 平均兄弟姐妹人数 | 平均父母孩子人数 |
| --- | --- | --- |
| 是 | 0.47 | 0.46 |
| 否 | 0.55 | 0.33 |

**分析**

- 分布比较平均，没看出太多信息
<a name="gzYPh"></a>
### 3.6 生还与票价关系
| 是否生还 | 平均票价 |
| --- | --- |
| 是 | 48.40 |
| 否 | 22.12 |

- 很明显可以看到，未生还里的人穷人明显要多，与3.2的结论是一致的。
<a name="Ou0TF"></a>
### 3.7 生还与船舱号关系
| Cabin | 生还率 |
| --- | --- |
| A | 0.47 |
| B | 0.74 |
| C | 0.59 |
| D | 0.76 |
| E | 0.75 |
| F | 0.62 |
| G | 0.5 |
| N | 0.3 |
| T | 0 |

<a name="lzWA5"></a>
### 3.8 生还与登船地点关系
![08_survive_embarked.png](https://cdn.nlark.com/yuque/0/2021/png/21513897/1627808612035-cbf7688e-b5c1-4ab2-8bbe-b4f0f860ed1d.png#height=480&id=tkuVj&margin=%5Bobject%20Object%5D&name=08_survive_embarked.png&originHeight=480&originWidth=640&originalType=binary&ratio=1&size=18276&status=done&style=none&width=640)

- S港口登船的人最多，死亡人数也最多。
<a name="oEGOe"></a>
## 4. 实验结果
将数据按照训练集和测试集1:5进行拆分，在测试集上的accuracy为**80.40%，**提交上kaggle后，得分为**0.76**。
