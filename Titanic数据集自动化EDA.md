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
