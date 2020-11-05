参照Kaggle上的https://www.kaggle.com/pedrodematos/titanic-a-complete-approach-to-top-rankings 进行探索性数据分析学习

基于Titanic Dataset进行自动化EDA， 不同数据的预处理和建模。主要目标是尝试提出一个完整的建模问题方法，从探索性数据分析到将监督和非监督学习技术应用于我们的数据。

## 1、数据探索

第一步是导入所需的库
开始分析之前让我们先看看数据集。为了节省时间在我们的探索数据分析过程，我们将使用2个库：pandas_profiling 和 autoviz。pandas_profile库在帮助我们了解正在研究的数据方面非常有用,进行Eda的过程节省时间，变得高效。

``` python
    """ 通过panas_profile自动生成报告，通过report.to_file将报告输出成html格式 “”“
    df = pd.read_csv("./Titanic/train.csv")
    report = pandas_profiling.ProfileReport(df)
    report.to_file(output_file="./Titanic/Report.html")
```

![输出报告](https://github.com/vivian315/KaggleEDA/blob/main/Report.html)

另一个功能强大的库是autoviz,使用此库，只需生成 1 行代码就生成多个图。与pandas_profiling相结合，少于5行代码，在几秒钟内就能获取大量信息。

```diff
+ 以下代码应该在jupyter notebook中运行，在pycharm中会报如下错误
- 运行中出现错误 NameError: name 'get_ipython' is not defined
```

``` python
    from autoviz.AutoViz_Class import AutoViz_Class
    av = AutoViz_Class()
    report = av.AutoViz(filename="./Titanic/train.csv")
```
    运行输出
    Shape of your Data Set: (891, 12)
    Classifying variables in data set...
        12 Predictors classified...
            This does not include the Target column(s)
        4 variables removed since they were ID or low-information variables
    Number of All Scatter Plots = 3
    Time to run AutoViz (in seconds) = 2.521

利用以上两个自动EDA库，我们可以从直方图到框图、关联矩阵等等图形化方式分别观察每个变量的行为。例如：
* 我们可以看到38%以上（341名）乘客幸存，同时 62%（549名）的乘客罹难
* Pclass列是乘客的舱位，它告诉我们其中55%(491名）的乘客在Class3，其中24%（216名）在class2，21%（184名）在class1。
* 数据集的乘客大多数为男性：约 35% （314名）乘客是女性，65% （577名）乘客是男性。
* age列中大约20%缺值。可以用各种技术来填充这些空位，例如用分布的均值填充它们。年龄分布有点偏斜，均值30岁左右，标准偏差接近15岁。在此数据集中乘客最大年龄是 80 岁。
* 根据"SibSP"栏目，大部分乘客（+68%）船上没有配偶或兄弟姐妹同行，Parch列也类似。
* 票价的分布更加偏斜。其平均值约为 32，标准差接近 50。其最小值为 0，最大值为 512.3292。这意味着，如果我们使用 SVM 等模型需谨慎处理此列。
* 在Embarked列展示72.3%的乘客在南安普敦港上船，18.9%的乘客在瑟堡港，8.6%的乘客在皇后镇港。
* 对于Pclass=1的乘客，其Fare较高 ，Pclass=2 次之，Pclass=3最低。从逻辑上讲，Pclass的分类是由乘客票价Fare的值定义的。
![](https://github.com/vivian315/KaggleEDA/blob/main/screenshots/p1.png?raw=true)
![](https://github.com/vivian315/KaggleEDA/blob/main/screenshots/pr1.png?raw=true)![](https://github.com/vivian315/KaggleEDA/blob/main/screenshots/pr2.png?raw=true)

### 更多探索
在进入建模部分之前，让我们先看看其它绘图，这些图形为我们提供与自动化EDA不同的视角。这可能给我们更多的启发，帮助我们了解在灾难中幸存下来的乘客和没有幸存下来的乘客之间的差异。
以下可视化效果，我们使用Plotly库完成

``` diff
+ * 首先，使用Violin图来查看幸存者与罹难者组年龄之间的差异
```

![](https://github.com/vivian315/KaggleEDA/blob/main/screenshots/P21.png?raw=true)

<details>
    <summary>点击展开代码</summary>
    
``` python 
    
        df = pd.read_csv("./Titanic/train.csv")
        df_survivors = df[df["Survived"] == 1]
        df_nonsurvivors = df[df["Survived"] == 0]

        # Violin 图填充数据
        violin_survivors = go.Violin(
            y=df_survivors["Age"],
            x=df_survivors["Survived"],
            name="Survivors",
            marker_color="forestgreen",
            box_visible=True)

        violin_nonsurvivors = go.Violin(
            y=df_nonsurvivors["Age"],
            x=df_nonsurvivors["Survived"],
            name="Non-Survivors",
            marker_color="darkred",
            box_visible=True)

        data = [violin_nonsurvivors, violin_survivors]

        # 设置背景色标题等
        layout = go.Layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title="幸存者年龄 vs 罹难者年龄",
            xaxis=dict(
                title="幸存否"
            ),
            yaxis=dict(
                title="年龄"
            )
        )

        fig = go.Figure(data=data, layout=layout)
        fig.show()
```
</details>

双侧检测

<details>
<summary>点击展开代码</summary>
    
``` python
    print("1、确定进行检验的假设（H0, H1），H0：幸存者与罹难者的年龄 没有显著差异，H1：幸存者与罹难者的年龄 有显著差异")
    print("2、选择检验统计量")
    df = pd.read_csv("./Titanic/train.csv")
    df_survivors = df[df["Survived"] == 1]
    df_nonsurvivors = df[df["Survived"] == 0]
    # 过滤年龄缺失的行
    dist = df["Age"].dropna()
    dist_a = df_survivors['Age'].dropna()
    dist_b = df_nonsurvivors['Age'].dropna()

    print("3、确定拒绝域 𝛂 = 0.05")
    print("4、求出检验统计量的P值")
    t_stat, p_value = stats.ttest_ind(dist_a, dist_b)
    print("\t\t----- T检测结果 -----")
    print("\t\tT stat. = " + str(t_stat))
    print("\t\tP value = " + str(p_value))  # P-value > 0.05，接受原假设，<0.05拒绝原假设
    print("5、查看样本结果")
    if p_value > 0.05:
        print("\t\tp值大于0.05接受H0拒绝H1 幸存者组与罹难者组年龄均值不存在显著差异")
    else:
        print("\t\tp值小于0.05拒绝H0接受H1，幸存者组与罹难者组年龄均值存在显著差异")
    print("")
```

</details>

结果：

    1、假设（H0, H1），H0：幸存者与罹难者的年龄 没有显著差异，H1：幸存者与罹难者的年龄 有显著差异
    2、选择检验统计量
    3、确定拒绝域 𝛂 = 0.05
    4、求出检验统计量的P值
         ----- T检测结果 -----
         T stat. = -2.06668694625381
         P value = 0.03912465401348249
    5、查看样本结果
         拒绝H0接受H1，幸存者组与罹难者组年龄均值存在显著差异
``` diff         
+ * 幸存者组和罹难者组的性别,pclass组成情况
```

![](https://github.com/vivian315/KaggleEDA/blob/main/screenshots/p22.png?raw=true)
![](https://github.com/vivian315/KaggleEDA/blob/main/screenshots/p23.png?raw=true)

<details>
    <summary>点击展开性别分析代码</summary>
    
``` python
    # 在幸存者中按性别计数
    df_survivors_sex = df_survivors["Sex"].value_counts()
    df_survivors_sex = pd.DataFrame({"Sex": df_survivors_sex.index, "count": df_survivors_sex.values})

    # 在罹难者中按性别计数
    df_nonsurvivors_sex = df_nonsurvivors["Sex"].value_counts()
    df_nonsurvivors_sex = pd.DataFrame({"Sex": df_nonsurvivors_sex.index, "count": df_nonsurvivors_sex.values})

    pie_survivors_sex = go.Pie(
        labels=df_survivors_sex["Sex"],
        values=df_survivors_sex["count"],
        domain=dict(x=[0, 0.5]),
        name="幸存者",
        hole=0.5,
        marker=dict(colors=["violet", "cornflowerblue"], line=dict(color="#000000", width=2))
    )

    pie_nonsurvivors_sex = go.Pie(
        labels=df_nonsurvivors_sex["Sex"],
        values=df_nonsurvivors_sex["count"],
        domain=dict(x=[0.5, 1.0]),
        name="罹难者",
        hole=0.5,
        marker=dict(colors=["cornflowerblue", "violet"], line=dict(color="#000000", width=2))
    )

    data = [pie_survivors_sex, pie_nonsurvivors_sex]

    layout = go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title="幸存者与罹难者的性别百分比",
        annotations=[dict(text="幸存者", x=0.18, y=0.5, font_size=15, showarrow=False),
                     dict(text="罹难者", x=0.85, y=0.5, font_size=15, showarrow=False)]
    )
    fig = go.Figure(data=data, layout=layout)
    fig.show()
```
</details>

<details>
    <summary>点击展开pclass分析代码</summary>

``` python

    # 幸存者组按Pclass计数
    df_survivors_pclass = df_survivors["Pclass"].value_counts()
    df_survivors_pclass = pd.DataFrame({"Pclass": df_survivors_pclass.index, "count": df_survivors_pclass.values})

    # 罹难者组按Pclass计数
    df_nonsurvivors_pclass = df_nonsurvivors["Pclass"].value_counts()
    df_nonsurvivors_pclass = pd.DataFrame(
        {"Pclass": df_nonsurvivors_pclass.index, "count": df_nonsurvivors_pclass.values})

    pie_survivors_pclass = go.Pie(
        labels=df_survivors_pclass["Pclass"],
        values=df_survivors_pclass["count"],
        domain=dict(x=[0, 0.5]),
        name="幸存者组",
        hole=0.5,
        marker=dict(colors=["#636EFA", "#EF553B", "#00CC96"], line=dict(color="#000000", width=2))
    )

    pie_nonsurvivors_pclass = go.Pie(
        labels=df_nonsurvivors_pclass["Pclass"],
        values=df_nonsurvivors_pclass["count"],
        domain=dict(x=[0.5, 1.0]),
        name="罹难者组",
        hole=0.5,
        marker=dict(colors=["#EF553B", "#00CC96", "#636EFA"], line=dict(color="#000000", width=2))
    )

    data = [pie_survivors_pclass, pie_nonsurvivors_pclass]

    layout = go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title="幸存者与罹难者的pclass百分比",
        annotations=[dict(text="幸存者组", x=0.18, y=0.5, font_size=15, showarrow=False),
                     dict(text="罹难者组", x=0.85, y=0.5, font_size=15, showarrow=False)]
    )

    fig = go.Figure(data=data, layout=layout)

    fig.show()
```
    
</details>

从上面显示的饼图中我们可以注意到：在罹难的乘客中近68%的乘客在Pclass3中。而幸存者中有近35%的乘客在Pclass3里。同样，幸存的乘客时，其中约40%的乘客在Pclass1。在罹难者中，只有14.6%的人处于Pclass1。看来Pclass与乘客在事故中幸存下来的事实之间有某种关系。让我们进一步探索

``` diff
+ * 幸存者组与罹难者组的船票价格分析
```

![](https://github.com/vivian315/KaggleEDA/blob/main/screenshots/p24.png?raw=true)
   
    Z检测与T检测结果
    ----- Z检测 -----    
    T stat. = 7.939191660871055
    P value = 2.035031103573989e-15

    ----- T检测 -----
    T stat. = 7.939191660871055
    P value = 6.120189341924198e-15

<details>
    <summary> 点击展开代码 </summary>
    
``` python

    fare_survivors_box = go.Box(
        y=df_survivors["Fare"],
        name="幸存者",
        marker=dict(color="navy")
    )

    fare_nonsurvivors_box = go.Box(
        y=df_nonsurvivors["Fare"],
        name="罹难者",
        marker=dict(color="steelblue")
    )

    data = [fare_nonsurvivors_box, fare_survivors_box]

    layout = go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title="幸存者与罹难者的【船票价格】对比",
        barmode="stack",
        yaxis=dict(
            title="船票价格分布"
        )
    )

    fig = go.Figure(data=data, layout=layout)
    fig.show()

    # 去掉Fare为空的行
    dist_c = df_survivors['Fare'].dropna()
    dist_d = df_nonsurvivors['Fare'].dropna()

    # Z-test: 检查分布均值（幸存者票价与非幸存者票价）在统计上是否不同
    t_stat_3, p_value_3 = ztest(dist_c, dist_d)
    print("----- Z检测 -----")
    print("T stat. = " + str(t_stat_3))
    print("P value = " + str(p_value_3))  # P-value< 0.05

    print("")

    # T-test: 检查分布均值（幸存者票价与非幸存者票价）在统计上是否不同
    t_stat_4, p_value_4 = stats.ttest_ind(dist_c, dist_d)
    print("----- T检测 -----")
    print("T stat. = " + str(t_stat_4))
    print("P value = " + str(p_value_4))  # P-value < 0.05
```
</details

查看票价的分布和假设检验，比较幸存者和罹难者我们可以再次观察到两组之间在统计学上存在显著差异。在查看箱线图时，我们可以看到与罹难者的票价值相比，幸存者的票价值通常较高。此信息可能与我们之前在饼图图上看到的"Pclass"百分比有关。

### PPS (Predictive Power Score)

您可能听说过相关矩阵。基本上，相关矩阵能够识别变量之间的线性关系。由于数据中的关系有时可能是非线性的（实际上大多数时候），我们可以使用 PPS（Predictive Power Score）矩阵来计算列之间的非线性关系。PPS 是一种非对称的、与数据类型无关的分数，可以检测两列之间的线性或非线性关系。分数范围从 0（无预测能力）到 1（完美预测能力）。它可以用作相关性（矩阵）的替代方法。

如果想了解为什么PPS很重要，建议你阅读这篇文章：https://towardsdatascience.com/rip-correlation-introducing-the-predictive-power-score-3d90808b9598
此外，请查阅 Python PPS 实现：https://github.com/8080labs/ppscore

<details>
    <summary> 点击展开代码 </summary>
    
``` python

    """ 以下代码在jupyter notebook 上运行"""
    df = pd.read_csv("./Titanic/train.csv")
    matrix_df = ppscore.matrix(df)[["x", "y", "ppscore"]].pivot(columns="x", index="y", values="ppscore")
    matrix_df = matrix_df.apply(lambda x: round(x, 2))  # Rounding matrix_df"s values to 0,XX
    sns.heatmap(matrix_df, vmin=0, vmax=1, cmap="Blues", linewidths=0.75, annot=True)
```
</details>

输出

![](https://github.com/vivian315/KaggleEDA/blob/main/screenshots/p25.png?raw=true)

查看此PPS矩阵，我们可以看到幸存变量的最佳单变量预测变量是列Ticket，0.19 pps ，其次是Sex，为0.13 pps 。这是有道理的， 因为妇女在救援过程中被优先考虑， 而且Ticket与Pclass密切相关。【Parch变量的最佳单变量预测变量是列Cabin，具有0.37pps】，等等。【】此处存疑？

## 2、监督学习：分类
我们使用的数据的上下文来深入了解建模部分。
### 特征工程（Feature Engineering）
为了帮助我们获得更好的性能，我们可以基于数据集的原始特征创建新要素。这里创建的一些新功能是基于Gunes Evitan在链接文章中的伟大想法：![Gunes Evitan的泰坦尼克号 - 高级特征工程教程](https://www.kaggle.com/gunesevitan/titanic-advanced-feature-engineering-tutorial#2.-Feature-Engineering)

<details>
    <summary> 点击展开代码 </summary>
    
``` python
# 关闭 chained_assignments,
# 否则出现警告"SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame
pd.options.mode.chained_assignment = None

df = pd.read_csv("./Titanic/train.csv")

# 创建一个AgeCat分类列，18岁以下是young，18到56岁是mature，56岁以上是senior
df['AgeCat'] = ''
df['AgeCat'].loc[(df['Age'] < 18)] = 'young'
df['AgeCat'].loc[(df['Age'] >= 18) & (df['Age'] < 56)] = 'mature'
df['AgeCat'].loc[(df['Age'] >= 56)] = 'senior'

# 创建 FamilySize分类列，小于2人是small，2到5人是medium，5人以上是large
df['FamilySize'] = ''
df['FamilySize'].loc[(df['SibSp'] <= 2)] = 'small'
df['FamilySize'].loc[(df['SibSp'] > 2) & (df['SibSp'] <= 5 )] = 'medium'
df['FamilySize'].loc[(df['SibSp'] > 5)] = 'large'

# 创建 IsAlone分类列，标注是否为单人
df['IsAlone'] = ''
df['IsAlone'].loc[((df['SibSp'] + df['Parch']) > 0)] = 'no'
df['IsAlone'].loc[((df['SibSp'] + df['Parch']) == 0)] = 'yes'

# 创建SexCat分类列标注 Young/Mature/Senior 男士还是 Young/Mature/Senior 女士
df['SexCat'] = ''
df['SexCat'].loc[(df['Sex'] == 'male') & (df['Age'] <= 21)] = 'youngmale'
df['SexCat'].loc[(df['Sex'] == 'male') & ((df['Age'] > 21) & (df['Age']) < 50)] = 'maturemale'
df['SexCat'].loc[(df['Sex'] == 'male') & (df['Age'] > 50)] = 'seniormale'
df['SexCat'].loc[(df['Sex'] == 'female') & (df['Age'] <= 21)] = 'youngfemale'
df['SexCat'].loc[(df['Sex'] == 'female') & ((df['Age'] > 21) & (df['Age']) < 50)] = 'maturefemale'
df['SexCat'].loc[(df['Sex'] == 'female') & (df['Age'] > 50)] = 'seniorfemale'


# 创建title分类列，Title列值由Name的前缀提取
# 另外创建Is_Married分类列，指明是否结婚，Mrstitle的幸存率高于其它女性title
df['Title'] = df['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
df['Is_Married'] = 0
df['Is_Married'].loc[df['Title'] == 'Mrs'] = 1
df['Title'] = df['Title'].replace(['Miss', 'Mrs','Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'], 'Miss/Mrs/Ms')
df['Title'] = df['Title'].replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'], 'Dr/Military/Noble/Clergy')


# 创建"Ticket Frequency"特征列
# 有太多的票值需要分析，因此按频率对它们进行分组会使事情更容易
df['Ticket_Frequency'] = df.groupby('Ticket')['Ticket'].transform('count')
```
</details>

创建新特征列后，可以删除在训练过程中不使用的无用的列。

<details>
    <summary> 点击展开代码 </summary>

``` python
    # 拆分目标
    target = df['Survived']

    # 删除无用列
    df.drop(['PassengerId', 'Survived', 'Ticket', 'Name', 'Cabin'], axis=1, inplace=True)

    df.to_csv("./Titanic/train_target.csv")
    # 拆分分类列和数值列
    categorical_df = df.select_dtypes(include=['object'])
    numeric_df = df.select_dtypes(exclude=['object'])

    # 存储分类列和数字列的名称。
    categorical_columns = list(categorical_df.columns)
    numeric_columns = list(numeric_df.columns)

    print("Categorical columns:\n", categorical_columns)
    print("\nNumeric columns:\n", numeric_columns)

    return target, categorical_columns, numeric_columns
```
</details>

### 平衡数据
正如我们在EDA部分看到的，train.csv的数据非常平衡，但我们的训练集中只有少量的观察。要尝试解决这个问题，我们可以采取不同的方法。三个常见方法是 RandomUnderSampling, SMOTE 和 SMOTEENN。我们可以尝试使用其中之一来平衡我们的数据。我们还可以选择不平衡数据，直接进入Pipeline部分。

### 模型训练和评估特征

经过所有预处理后，我们现在已准备好构建和评估不同的机器学习模型。首先，让我们创建一个函数，负责在稍后将创建的测试集上评估分类器。

## 管道构建
什么是管道？
我们可以将管道理解为数据中应用的操作序列，一条完整的管道是由几个不同的小管道链接而成。将此应用到数据科学：假设每个小管道都是建模过程中的步骤。例如：
Step1：填充数值列中null值。
Step2：规范化数值要素，使之处于相同的比例。
Step3：填充分类要素中的null值。
Step4：OneHotEncode分类功能。
Step5：拟合机器学习模型并对其进行评估。

我们可以创建一个管道对象来统一所有这些步骤，然后将该对象放入我们的训练数据中，而不是单独执行以上每个步骤。

我们为什么要那么做？使用管道有很多优点。
* 1 - 生产代码更容易实现
将机器学习模型部署到生产环境中时，主要目标是将其用于以前从未见过的数据。为此，需要像训练集那样转换新数据。对于每个预处理任务，可以使用单个管道对象按顺序应用所有函数，而不是为每个预处理任务应用多个不同的函数。这意味着，在一行代码中，您可以应用所有所需的转换。在此笔记本的"预测"部分中查看此示例。
* 2 - 与RandomSearchCV结合使用时，可以测试多个不同的管道选项
在训练模型时，您一定已经问过自己："对于这种类型的数据，什么效果最好？用列的平均值或中位数填充缺失的值？我应该使用MinMaxScaler还是StandardScale？应用尺寸减小？创建更多特征（例如PolynomialFeatures）？使用管道和超参数搜索函数（如 RandomSearchCV），您可以自动搜索整个数据管道、模型和参数集，从而节省您用于搜索最佳工程方法和模型/超参数所投入的精力。
假设我们有 4 个不同的管道：

Pipeline1：通过计算每列的平均值来填充数字要素中缺失的值 - 应用MinMaxScaler - 将 OneHotEncoder 应用于分类要素 - 将数据拟合到具有 n_neighbors = 15 的 KNN 分类器中。
Pipeline2：通过计算每列的平均值来填充数字要素中的缺失值 - 应用StandardScaler - 将 OneHotEncoder 应用于分类要素 - 将数据拟合到具有 n_neighbors = 30 的 KNN 分类器中。
Pipeline3：通过计算每列的中位数来填充数字要素中的缺失值 - 应用MinMaxScaler - 将 OneHotEncoder 应用于分类要素 - 将数据拟合到具有 n_estimators = 100 的随机林分类器中。
Pipeline4：通过计算每列的中位数来填充数字要素中的缺失值 - 应用StandardScaler - 将 OneHotEncoder 应用于分类要素 - 将数据拟合到具有 n_estimators = 150 的随机林分类器中。

开始可能会认为要检查哪个Pipeline更好只需要手动创建所有这些管道，拟合数据然后评估结果。但是，如果我们想要增加这个搜索的范围，比如超过数百个不同的Pipeline呢？手动执行此操作真的很难。这就是 RandomSearchcv 发挥作用的地方。

* 3 - 交叉验证时无信息泄漏
这个有点棘手，特别对于初学者。基本上在交叉验证时数据应被转换到每个CV步骤中，而不是以前。转换训练集（例如使用StandardScaler）后执行交叉验证时，来自该集的信息将泄露到验证集。这可能会导致偏颇/不理想的结果。正确的做法是在交叉验证中规范化数据。这意味着对于每个CV步骤，一个Scaler仅适配到训练集上。然后，此Scaler转换验证集并评估模型。这样，训练集中的信息不会泄露到验证集。当使用RandomSearchCV（或GridSearchCV）内的Pipeline时，此问题将得到处理。
这是机器学习中的一个关键概念，因此了解原因非常重要。建议阅读有关该主题的更深入的文章。此外，Andreas C. Muller & Sarah Guido 的《Python 机器学习导论》一书的第 6 章（主要是第 306 页和 307 页）给出了对这个问题的很好的看法。
关于Pipelines和RandomSearchCV更多的信息请参照:![Pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
![RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)




