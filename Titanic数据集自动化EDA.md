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
- 运行中出现错误 NameError: name 'get_ipython' is not defined，暂时诸事掉AutoViz_Class.py的第37，以后查找解决办法
```

    Shape of your Data Set: (891, 12)
    Classifying variables in data set...
        12 Predictors classified...
            This does not include the Target column(s)
        4 variables removed since they were ID or low-information variables
    Number of All Scatter Plots = 3
    Time to run AutoViz (in seconds) = 2.521
