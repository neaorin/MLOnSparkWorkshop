# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Databricks - Credit Scoring
# MAGIC 
# MAGIC ## Introduction
# MAGIC 
# MAGIC Banks play a crucial role in market economies. They decide who can get finance and on what terms and can make or break investment decisions. For markets and society to function, individuals and companies need access to credit. 
# MAGIC 
# MAGIC Credit scoring algorithms, which make a guess at the probability of default, are the method banks use to determine whether or not a loan should be granted. 
# MAGIC 
# MAGIC ## The problem
# MAGIC 
# MAGIC Down below you will find a possible solution to the challenge described in [c/GiveMeSomeCredit](https://www.kaggle.com/c/GiveMeSomeCredit) where participants where required to improve on the state of the art in credit scoring, by predicting the probability that somebody will experience financial distress in the next two years. 
# MAGIC 
# MAGIC ## The data
# MAGIC 
# MAGIC The training data contains the following variables:
# MAGIC 
# MAGIC 
# MAGIC | **Variable   Name**                  | **Description**                                                                                                                                              | **Type**   |
# MAGIC |--------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|
# MAGIC | SeriousDlqin2yrs                     | Person   experienced 90 days past due delinquency or worse                                                                                                   | *Y/N*      |
# MAGIC | RevolvingUtilizationOfUnsecuredLines | Total   balance on credit cards and personal lines of credit except real estate and   no installment debt like car loans divided by the sum of credit limits | percentage |
# MAGIC | age                                  | Age of borrower in   years                                                                                                                                   | integer    |
# MAGIC | NumberOfTime30-59DaysPastDueNotWorse | Number of times   borrower has been 30-59 days past due but no worse in the last 2 years.                                                                    | integer    |
# MAGIC | DebtRatio                            | Monthly debt   payments, alimony,living costs divided by monthy gross income                                                                                 | percentage |
# MAGIC | MonthlyIncome                        | Monthly income                                                                                                                                               | real       |
# MAGIC | NumberOfOpenCreditLinesAndLoans      | Number of Open loans   (installment like car loan or mortgage) and Lines of credit (e.g. credit   cards)                                                     | integer    |
# MAGIC | NumberOfTimes90DaysLate              | Number of times   borrower has been 90 days or more past due.                                                                                                | integer    |
# MAGIC | NumberRealEstateLoansOrLines         | Number of mortgage   and real estate loans including home equity lines of credit                                                                             | integer    |
# MAGIC | NumberOfTime60-89DaysPastDueNotWorse | Number of times   borrower has been 60-89 days past due but no worse in the last 2 years.                                                                    | integer    |
# MAGIC | NumberOfDependents                   | Number of dependents   in family excluding themselves (spouse, children etc.)                                                                                | integer    |
# MAGIC 
# MAGIC The **SeriousDlqin2yrs** is the dependent variable of the dataset, or better named the **label**. This is a boolean value which details if a certain individual has experienced a deliquency of 90 days past due or worse in the last 2 years.
# MAGIC 
# MAGIC You can get the training data from [here](https://github.com/dlawrences/GlobalAINightBucharest/blob/master/data/cs-training.csv).
# MAGIC 
# MAGIC This dataset should be used for:
# MAGIC - creating two smaller sets, one for the actual training (e.g. 80%) and one for testing (e.g. 20%)
# MAGIC - during cross validation, if you want to do the validation on multiple different folds of data to manage better the bias and the variance
# MAGIC 
# MAGIC The benchmark/real unseen data you could use to test your model predictions may be downloaded from [here](https://github.com/dlawrences/GlobalAINightBucharest/blob/master/data/cs-test.csv).
# MAGIC 
# MAGIC ## The Data Science Process
# MAGIC 
# MAGIC This is the outline of the process we'll be following in this workshop.
# MAGIC 
# MAGIC ![Data Science Process](https://raw.githubusercontent.com/neaorin/PredictTheWorldCup/master/images/datascience_process.jpg)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data import
# MAGIC 
# MAGIC Before starting to do anything else, we need to import the data. First, let's download both datasets and store them in DBFS.

# COMMAND ----------

import urllib.request

training_data_url = "https://raw.githubusercontent.com/dlawrences/GlobalAINightBucharest/master/data/cs-training.csv"
training_data_filename = "cs_training.csv"

test_data_url = "https://raw.githubusercontent.com/dlawrences/GlobalAINightBucharest/master/data/cs-test.csv"
test_data_filename = "cs_test.csv"

dbfs_data_folder = "dbfs/FileStore/data/"
project_name = 'credit-scoring'

dbfs_project_folder = dbfs_data_folder + project_name + "/"

# Download files and move them to the final directory in DBFS
urllib.request.urlretrieve(training_data_url, "/tmp/" + training_data_filename)
urllib.request.urlretrieve(test_data_url, "/tmp/" + test_data_filename)

# Create the project directory if it does not exist and move files to it
dbutils.fs.mkdirs(dbfs_project_folder)
dbutils.fs.mv("file:/tmp/" + training_data_filename, dbfs_project_folder)
dbutils.fs.mv("file:/tmp/" + test_data_filename, dbfs_project_folder)

# List the contents of the directory
dbutils.fs.ls(dbfs_project_folder)

# COMMAND ----------

import numpy as np # library for linear algebra and stuff
import pandas as pd # library for data processing, I/O on csvs etc
import matplotlib.pyplot as plt # library for plotting
import seaborn as sns # a library which is better for plotting

# File location and type
file_location = dbfs_project_folder + training_data_filename
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
creditSDF = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(creditSDF)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC Now that we've loaded the dataset, the first thing we're going to do is set aside a part of it - 25 to 30 percent is a usual percentage - and not touch it until it's time to test our models.

# COMMAND ----------

temp_table_name = "trainingSDF"

# Split the data into training and test sets (25% held out for testing)
(trainingSDF, testingSDF) = creditSDF.randomSplit([0.75, 0.25], seed=1)

# Make the dataframe available in the SQL context
trainingSDF.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

# Sample out 10 rows of the dataset
display(trainingSDF.sample(False, 0.1, seed=0).limit(10))

# COMMAND ----------

# Inspect the schema
trainingSDF.printSchema()

# COMMAND ----------

# Check of the summary statistics of the features
display(trainingSDF.describe())

# COMMAND ----------

# highlight how many missing values we have for every feature
from pyspark.sql.functions import lit, col

rows = trainingSDF.count()
summary = trainingSDF.describe().filter(col("summary") == "count")
display(summary.select(*((lit(rows)-col(c)).alias(c) for c in trainingSDF.columns)))

# COMMAND ----------

# MAGIC %md
# MAGIC Quick conclusions:
# MAGIC - there are a lot of null values for **MonthlyIncome** and **NumberOfDependents**; we will analyse how to impute these next
# MAGIC - the minimum value for the **age** variable is 0 and it presents an outlier/bad data; this will be imputed with the median
# MAGIC - the maximum value of **329664** for the **DebtRatio** variable is rather weird given this variable is a mere percentage; from a modelling perspective, thus we will need to understand why there are such big values and decide what to do with them
# MAGIC - the maximum value of **50708** for the **RevolvingUtilizationOfUnsecuredLines** variable is rather weird given this variable is a mere percentage; rom a modelling perspective, thus we will need to understand why there are such big values and decide what to do with them

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exploratory Data Analysis & Data Cleaning
# MAGIC 
# MAGIC We are going to take step by step most of the interesting columns that need visualizing and cleansing to be done.
# MAGIC 
# MAGIC ### Target class - SeriousDlqin2yrs
# MAGIC 
# MAGIC Let's understand the distribution of our target class (**SeriousDlqin2yrs**). This could very well influence the algorithm we will want to use to model the problem.

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select SeriousDlqin2yrs, count(*) as TotalCount from trainingSDF group by SeriousDlqin2yrs

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC There seems to be a lot of **class imbalance** going on around here. Let's understand the positive event rate in our target class.

# COMMAND ----------

class_0 = trainingSDF.filter(trainingSDF.SeriousDlqin2yrs == 0).count()
class_1 = trainingSDF.filter(trainingSDF.SeriousDlqin2yrs == 1).count()

print("Total number of observations with a class of 0: {}".format(class_0))
print("Total number of observations with a class of 1: {}".format(class_1))
print("Positive event rate: {} %".format(class_1/(class_0+class_1) * 100))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC A positive event rate of 6.6% is by no means ideal. Going through with this distribution for the target class may mean that the minorit class will be ignored by the algorithm we are going to use to model the problem, thus the model will be biased to customers which are not likely to default.
# MAGIC 
# MAGIC A couple of ideas which we are going to take into consideration going further to go around this problem:
# MAGIC - given we have a lot of training data (100k+ observations), we may actually considering resampling the dataset.
# MAGIC - we are going to use an evaluation metric which compensates the imbalance between classes, e.g. **ROC AUC**

# COMMAND ----------

# MAGIC %md
# MAGIC ### Age variable
# MAGIC 
# MAGIC We are interested in knowing the distribution of the **age** variable. Ideally, we would want this to be a normal distribution altogether.
# MAGIC 
# MAGIC We are also not looking for customers under the legal age of 18 years. If any, we will impute the age of these with the median of the column.

# COMMAND ----------

import matplotlib.ticker as ticker

# spark.sql does not have any histogram method, however the RDD api does
age_histogram = trainingSDF.select('age').rdd.flatMap(lambda x: x).histogram(10)

fig, ax = plt.subplots()

# the computed histogram needs to be loaded in a pandas dataframe so we will be able to plot it using sns
age_histogram_df = pd.DataFrame(
    list(zip(*age_histogram)), 
    columns=['bin', 'frequency']
)

ax = sns.barplot(x = "bin", y = "frequency", data = age_histogram_df)

ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(age_histogram_df.iloc[x]['bin'], '.1f')))

display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC It seems there may be customers under the legal age. Let's see how many.

# COMMAND ----------

# We can use the filter method to understand what are the observations for which the customers falls under the legal age.
display(trainingSDF.filter(trainingSDF.age < 18))

# COMMAND ----------

# MAGIC %md
# MAGIC Fortunately there is only one. Let's impute this value with the median.

# COMMAND ----------

# Import functions which will help us code an if statement
from pyspark.sql import functions as F

def imputeAgeWithMedian(df, medianAge):

  # Update with the median for the rows where the age columnis equal to 0
  df = df.withColumn('age',
                                       F.when(
                                           F.col('age') == 0,
                                           medianAge
                                       ).otherwise(
                                           F.col('age')
                                       )
                )
  
  return df

# Compute the median of the age variable
trainingMedianAge = np.median(trainingSDF.select('age').dropna().collect())
trainingSDF = imputeAgeWithMedian(trainingSDF, trainingMedianAge)

# Check to see that the only row shown above has a new age value
display(trainingSDF.filter(trainingSDF.Idx == 65696))

# COMMAND ----------

# MAGIC %md
# MAGIC Finally, let's check the distribution of the age for each group, based on the values for the **SeriousDlqin2yrs** target variable.
# MAGIC 
# MAGIC We're going to use a [box and whiskers plot](https://towardsdatascience.com/understanding-boxplots-5e2df7bcbd51?gi=9e6b6042f263) to better visualize the distribution.

# COMMAND ----------

fig, ax = plt.subplots()

ax = sns.boxplot(x="SeriousDlqin2yrs", y="age", data = trainingSDF.toPandas())

display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC Based on the cleaned age column, let's create an age banding column (bins) which correlates better to credit risk.
# MAGIC 
# MAGIC For this example, we are going to use the bins included in this paper: [figure in paper](https://www.researchgate.net/figure/Percentage-of-default-risk-among-different-age-groups_fig2_268345909)

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

def bandingFunction(age):
  if (age < 25):
    return '18-25'
  elif (age >= 25 and age < 30): 
    return '25-29'
  elif (age >= 30 and age < 35):
    return '30-34'
  elif (age >= 35 and age < 40):
    return '35-39'
  elif (age >= 40 and age < 45):
    return '40-44'
  elif (age >= 45 and age < 50):
    return '45-49'
  elif (age >= 50 and age < 55):
    return '50-54'
  elif (age >= 55 and age < 60):
    return '55-59'
  elif (age >= 60 and age < 65):
    return '60-64'
  elif (age >= 65 and age < 70):
    return '65-69'
  elif (age >= 70 and age < 75):
    return '70-74'
  elif (age >= 75): 
    return '75+'
  else: 
    return ''

age_banding_udf = udf(bandingFunction, StringType() )

def addAgeBanding(df):
  df = df.withColumn('age_banding', age_banding_udf(df.age))
  return df.drop('age')

trainingSDF = addAgeBanding(trainingSDF)

trainingSDF.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's now visualize the distribution.
# MAGIC 
# MAGIC NOTE: as an alternative to Python-based plotting libraries like *seaborn* or *pyplot* we can also use Databricks' built-in visualizations. Click on the Visualization button below the results of this cell to select a **Bar** chart.

# COMMAND ----------

# MAGIC %sql
# MAGIC select age_banding, count(*) as Counts from trainingSDF group by age_banding order by age_banding

# COMMAND ----------

# MAGIC %md
# MAGIC ### MonthlyIncome variable
# MAGIC 
# MAGIC In credit scoring, the income of the individual - besides the other debt that he is into - is of greater importance than other things when it comes to the final decision.
# MAGIC 
# MAGIC Let's see how the distribution of this variable looks.

# COMMAND ----------

fig, ax = plt.subplots()

ax = sns.boxplot(x="SeriousDlqin2yrs", y="MonthlyIncome", data = trainingSDF.toPandas())

display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Hmm, the chart isn't that useful, probably because we have some very large outliers (large *MonthlyIncome* values) skewing the plot. Let's try using a log scale for the y axis:

# COMMAND ----------

fig, ax = plt.subplots()

ax = sns.boxplot(x="SeriousDlqin2yrs", y="MonthlyIncome", data = trainingSDF.toPandas())
ax.set_yscale("log")

display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC That's better. One thing is certain - people which have gone through issues usually have a lower income. However, it looks like the dataset contains really low values - like 5$ or less a month which is really odd.
# MAGIC 
# MAGIC For our reference, let's view the [Characteristics of Minimum Wage Workers in the US: 2010](https://www.bls.gov/cps/minwage2010.htm). In this article, it is stated that the prevailing Federal minimum wage was $7.25 per hour.
# MAGIC 
# MAGIC In this case, considering an individual would work on a full-time basis for 52 weeks straight in a year, that individual would earn **$7.25 X 40 hrs X 52 weeks** = **_$15,080_**.
# MAGIC 
# MAGIC This translates to approximately **_$1,256_** a month. For a part-time worker, this would mean a wage of **_$628_**. For an individual working only a quarter of the total time, that would mean a wage of only **_$314_**.
# MAGIC 
# MAGIC According to the [US Census Bureau, Current Population Survey 2016](https://en.wikipedia.org/wiki/Personal_income_in_the_United_States#cite_note-CPS_2015-2), 6.48% of people earned **_$2,500_** or less in a full year. This translates to only **_$208_** a month. Median personal income comes to about **_$31,099_** a year, which is about **_$2,592_** dollars a month.
# MAGIC 
# MAGIC Given all this information, let's do some more exploratory data analysis to see where this odd **MonthlyIncome** needs patching a bit.

# COMMAND ----------

# MAGIC %md
# MAGIC Off the bat, there is weirdness in having NULL **MonthlyIncome** data, but being able to calculate **DebtRatio**.

# COMMAND ----------

# MAGIC %sql
# MAGIC select avg(DebtRatio), count(1) as Instances from trainingSDF where MonthlyIncome is null

# COMMAND ----------

# MAGIC %md
# MAGIC It may be the case than whoever gathered this data have replaced **NULL** in this column to 1 to be able to calculate the **DebtRatio** using the data about the **TotalDebt** of the individual they had. This will be need to be treated this way:
# MAGIC - impute the **NULL** values with the median of the dataset
# MAGIC - recalculate the **DebtRatio** given we know that the **TotalDebt** is currently equal for those individuals to the value of the **DebtRatio**

# COMMAND ----------

# MAGIC %md
# MAGIC A very low **MonthlyIncome** between $1 and $7 is again a bit suspicious (having worked under 1hr per month). Let's see a list of people with very small monthly incomes:

# COMMAND ----------

# MAGIC %sql
# MAGIC select MonthlyIncome, count(1) as Instances, avg(DebtRatio) from trainingSDF where MonthlyIncome between 1 and 100 group by MonthlyIncome order by 1

# COMMAND ----------

# MAGIC %md
# MAGIC Given the number of records where **MonthlyIncome** is equal to 1 is suspiciously high, we are going to impute it like we do for the **NULL** values. However, for the other values, there isn't just too much wrong data to draw any conclusions. If we extend the window up to 208:

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(1) as Instances from trainingSDF where MonthlyIncome between 2 and 208

# COMMAND ----------

# MAGIC %md
# MAGIC 147 rows is a low percentage of samples from the whole dataset, so we will be keeping these as they are.

# COMMAND ----------

# MAGIC %md
# MAGIC That's quite a lot of information, so let's wrap up what we are going to do:
# MAGIC 
# MAGIC For the specifics of this lab, we are going to consider that:
# MAGIC - observations with a MonthlyIncome of 1 will be processed to get the median MonthlyIncome
# MAGIC - observations with a MonthlyIncome of null will be processed to get the median MonthlyIncome
# MAGIC 
# MAGIC Given the **DebtRatio** has been computed as the overall **Debt** divided by the **MonthlyIncome**, we are going to regenerate the initial debt first so we can use it later to recompute the **DebtRatio** based on the then cleaned **MonthlyIncome**.

# COMMAND ----------

# MAGIC %md
# MAGIC First, we save the initial **Debt** so we are able to recompute the updated DebtRatio afterwards.

# COMMAND ----------

from pyspark.sql import functions as F

def addInitialDebtColumn(df):
  df = df.withColumn(
                  'initialDebt',
                  F.when(
                      (((F.col('MonthlyIncome') >= 0) & (F.col('MonthlyIncome') <= 1)) | (F.col('MonthlyIncome').isNull())),
                      F.col('DebtRatio')
                  ).otherwise(
                      F.col('MonthlyIncome') * F.col('DebtRatio')
                  )
              )
  
  return df
  
trainingSDF = addInitialDebtColumn(trainingSDF)

# COMMAND ----------

display(trainingSDF)

# COMMAND ----------

# MAGIC %md
# MAGIC After the initial **Debt** has been saved, we are good to start imputing the **MonthlyIncome** column. 
# MAGIC If the actual value is <= $7 or missing, we manually impute using the **numpy**-calculated median.

# COMMAND ----------

def imputeMonthlyIncome(df, incomeMedian):
  # Apply income median if the MonthlyIncome is <=7, or null
  
  df = df.withColumn('MonthlyIncome',
                                       F.when(
                                           (((F.col('MonthlyIncome') >= 0) & (F.col('MonthlyIncome') <= 7)) | (F.col('MonthlyIncome').isNull())),
                                           incomeMedian
                                       ).otherwise(
                                           F.col('MonthlyIncome')
                                       )
                )
  
  return df

trainingIncomeMedian = np.median(trainingSDF.select('MonthlyIncome').dropna().collect())
trainingSDF = imputeMonthlyIncome(trainingSDF, trainingIncomeMedian)

# COMMAND ----------

# MAGIC %md
# MAGIC Now that the **MonthlyIncome** variable has been imputed, let's recalculate a more correct **DebtRatio** based on the initial **Debt** we have saved previously.

# COMMAND ----------

def recalculateDebtRatio(df):
  df = df.withColumn(
                    'DebtRatio',
                    df.initialDebt/df.MonthlyIncome
                )
  
  return df

trainingSDF = recalculateDebtRatio(trainingSDF)

trainingSDF.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's see how many values in this column are actually exceeding the threshold of **1** now.

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(1) from trainingSDF where DebtRatio > 1

# COMMAND ----------

# MAGIC %md
# MAGIC From **35137** records down to this. Let's see how it looks from a distribution point of view.

# COMMAND ----------

fig, ax = plt.subplots()

ax = sns.boxplot(x="DebtRatio", data = trainingSDF.toPandas())
ax.set_xscale("log")

display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC It seems this values are going up into the hundreds. Individuals may exceed a **DebtRatio** of 1 whenever they are lending more than they are earning (and some people in difficult scenarios tend to do that).
# MAGIC 
# MAGIC Let's default the higher values to a threshold of **1.5**.

# COMMAND ----------

def defaultDebtRatioToThreshold(df):
  df = df.withColumn('DebtRatio',
                                       F.when(
                                           (F.col('DebtRatio') > 1.5),
                                           1.5
                                       ).otherwise(
                                           F.col('DebtRatio')
                                       )
                )
  
  return df

trainingSDF = defaultDebtRatioToThreshold(trainingSDF)

trainingSDF.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### RevolvingUtilizationOfUnsecuredLines variable
# MAGIC Let's understand how many values exceed 1 for this column and default them to this max value.

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(1) from trainingSDF where RevolvingUtilizationOfUnsecuredLines > 1

# COMMAND ----------

# MAGIC %md
# MAGIC Some records have a **RevolvingUtilizationOfUnsecuredLines** value higher than 1. Given the total balance on credit cards and personal lines of credit is divided to the sum of credit limits, this should not exceed 1.
# MAGIC 
# MAGIC Let's view the distribution of it and then default the weird records to this threshold.

# COMMAND ----------

fig, ax = plt.subplots()

ax = sns.boxplot(x="RevolvingUtilizationOfUnsecuredLines", data = trainingSDF.toPandas())
ax.set_xscale("log")


display(fig)

# COMMAND ----------

def defaultRevolvingUtilizationToThreshold(df):
  df = df.withColumn('RevolvingUtilizationOfUnsecuredLines',
                                       F.when(
                                           (F.col('RevolvingUtilizationOfUnsecuredLines') > 1),
                                           1
                                       ).otherwise(
                                           F.col('RevolvingUtilizationOfUnsecuredLines')
                                       )
                )
  
  return df

trainingSDF = defaultRevolvingUtilizationToThreshold(trainingSDF)

trainingSDF.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### NumberOfDependents variable
# MAGIC 
# MAGIC Let's understand how many missing values this column has.

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(1) from trainingSDF where NumberOfDependents is null

# COMMAND ----------

# MAGIC %md
# MAGIC About 3000 missing values out of the total number of rows is not bad at all.
# MAGIC 
# MAGIC Let's see how the distribution of this variable looks. We will understand the mode from it and will be able to impute using it.

# COMMAND ----------

# spark.sql does not have any histogram method, however the RDD api does
dependents_histogram = trainingSDF.select('NumberOfDependents').rdd.flatMap(lambda x: x).histogram(10)

fig, ax = plt.subplots()

# the computed histogram needs to be loaded in a pandas dataframe so we will be able to plot it using sns
dependents_histogram_df = pd.DataFrame(
    list(zip(*dependents_histogram)), 
    columns=['bin', 'count']
)

ax = sns.barplot(x = "bin", y = "count", data = dependents_histogram_df)

display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC We can tell from the barplot above that the mode (most frequent value) of this column is 0. Let's impute the missing values with it.

# COMMAND ----------

def imputeNumberOfDependents(df):
  df = df.withColumn('NumberOfDependents',
                                       F.when(
                                           (F.col('NumberOfDependents').isNull()),
                                           0
                                       ).otherwise(
                                           F.col('NumberOfDependents')
                                       )
                )

  return df

trainingSDF = imputeNumberOfDependents(trainingSDF)
  
trainingSDF.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

# Check of the summary statistics of the features now
display(trainingSDF.describe())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Building our first model

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import VectorAssembler, StringIndexer, MinMaxScaler

# Index categorical features
categorical_indexer = StringIndexer(inputCol="age_banding", outputCol="age_banding_indexed")

# assemble all features into a features vector
feature_assembler = VectorAssembler(
    inputCols=[
   'RevolvingUtilizationOfUnsecuredLines',
   'NumberOfTime30-59DaysPastDueNotWorse',
   'NumberOfOpenCreditLinesAndLoans',
   'NumberOfTimes90DaysLate',
   'NumberRealEstateLoansOrLines',
   'NumberOfTime60-89DaysPastDueNotWorse',
   'NumberOfDependents',
   'age_banding_indexed',
   'initialDebt',
   'DebtRatio',
   'MonthlyIncome'],
    outputCol="features")

# scale features 
scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")


# Train a DecisionTree model.
decision_tree_classifier = DecisionTreeClassifier(labelCol="SeriousDlqin2yrs", featuresCol="scaledFeatures",
                            impurity="gini", maxDepth=5, seed=1)

# Chain assembler and model in a Pipeline
dtc_pipeline = Pipeline(stages=[categorical_indexer, feature_assembler, scaler, decision_tree_classifier])

# Train model. 
dtc_model = dtc_pipeline.fit(trainingSDF)

print(dtc_model.stages[3])

# COMMAND ----------

# MAGIC %md
# MAGIC # Testing the model

# COMMAND ----------

testingSDF = imputeAgeWithMedian(testingSDF, trainingMedianAge)
testingSDF = addAgeBanding(testingSDF)
testingSDF = addInitialDebtColumn(testingSDF)
testingSDF = imputeMonthlyIncome(testingSDF, trainingIncomeMedian)
testingSDF = recalculateDebtRatio(testingSDF)
testingSDF = defaultDebtRatioToThreshold(testingSDF)
testingSDF = defaultRevolvingUtilizationToThreshold(testingSDF)
testingSDF = imputeNumberOfDependents(testingSDF)

# Make the dataframe available in the SQL context
test_temp_table_name = "testingSDF"

# Make the dataframe available in the SQL context
testingSDF.createOrReplaceTempView(test_temp_table_name)

display(testingSDF)

# COMMAND ----------

# Make predictions.
dtc_predictions = dtc_model.transform(testingSDF)

# Select example rows to display.
display(dtc_predictions.select("probability", "prediction", "SeriousDlqin2yrs"))


# COMMAND ----------

# display the confusion matrix
from sklearn.metrics import confusion_matrix

def plotConfusionMatrix(confusion_matrix):
  fig, ax = plt.subplots()
  plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Wistia)
  classNames = ['Negative','Positive']
  ax.set_title(f'Confusion Matrix')
  ax.set_ylabel('True label')
  ax.set_xlabel('Predicted label')
  tick_marks = np.arange(len(classNames))
  ax.set_xticks(tick_marks)
  ax.set_yticks(tick_marks)
  s = [['TN','FP'], ['FN', 'TP']]
  for i in range(2):
      for j in range(2):
          ax.text(j,i, str(s[i][j])+" = "+str(confusion_matrix[i][j]))
  display(fig)
  
  
dtc_confusion_matrix = confusion_matrix(dtc_predictions.select("SeriousDlqin2yrs").collect(), dtc_predictions.select("prediction").collect())
plotConfusionMatrix(dtc_confusion_matrix)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ###Precision and Recall
# MAGIC 
# MAGIC ![Precision and Recall](https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/Precisionrecall.svg/350px-Precisionrecall.svg.png)

# COMMAND ----------

tn, fp, fn, tp = dtc_confusion_matrix.ravel()
print(f"Precision = TP / (TP + FP) = {tp/(tp+fp)}")
print(f"Recall = TP / (TP + FN) = {tp/(tp+fn)}")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Sensitivity and Specificity, and the ROC Curve
# MAGIC 
# MAGIC ![Sensitivity and specificity](https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/Sensitivity_and_specificity.svg/350px-Sensitivity_and_specificity.svg.png)

# COMMAND ----------

# plot the ROC curve
from sklearn.metrics import roc_curve, auc

def plotROCCurve(predictions, show_thresholds=False):
  results = predictions.select(['probability', 'SeriousDlqin2yrs']).collect()
  y_score = [float(i[0][1]) for i in results]
  y_true = [float(i[1]) for i in results]

  fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label = 1)
  roc_auc = auc(fpr, tpr)

  fig, ax = plt.subplots()
  ax.plot(fpr, tpr, label='ROC curve (area = %0.4f)' % roc_auc)
  ax.plot([0, 1], [0, 1], 'k--')
  if show_thresholds:
      tr_idx = np.arange(385, len(thresholds), 700)
      for i in tr_idx:
        ax.plot(fpr[i], tpr[i], "xr")
        ax.annotate(xy=(fpr[i], tpr[i]), s="%0.3f" % thresholds[i])
  ax.set_xlim([0.0, 1.0])
  ax.set_ylim([0.0, 1.0])
  ax.set_xlabel('False Positive Rate (1 - Specificity)')
  ax.set_ylabel('True Positive Rate (Sensitivity)')
  ax.set_title('Receiver operating characteristic')
  ax.legend(loc="lower right")
  display(fig)

plotROCCurve(dtc_predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Gradient Boosted Trees

# COMMAND ----------

from pyspark.ml.classification import GBTClassifier


# Train a Gradient-boosted tree classifier model.
gbt_classifier = GBTClassifier(labelCol="SeriousDlqin2yrs", featuresCol="features",
                            maxIter=35, seed=1)

# Chain assembler and model in a Pipeline
gbt_pipeline = Pipeline(stages=[categorical_indexer, feature_assembler, scaler, gbt_classifier])

# Train model. 
gbt_model = gbt_pipeline.fit(trainingSDF)

print(gbt_model.stages[3])

# COMMAND ----------

# Make predictions.
gbt_predictions = gbt_model.transform(testingSDF)

# Select example rows to display.
display(gbt_predictions.select("probability", "prediction", "SeriousDlqin2yrs"))

# COMMAND ----------

gbt_confusion_matrix = confusion_matrix(gbt_predictions.select("SeriousDlqin2yrs").collect(), gbt_predictions.select("prediction").collect())
plotConfusionMatrix(gbt_confusion_matrix)

# COMMAND ----------

tn, fp, fn, tp = gbt_confusion_matrix.ravel()
print(f"Precision = TP / (TP + FP) = {tp/(tp+fp)}")
print(f"Recall = TP / (TP + FN) = {tp/(tp+fn)}")

# COMMAND ----------

# plot the precision - recall curve
from sklearn.metrics import precision_recall_curve

def plotPRCurve(predictions):
  results = predictions.select(['probability', 'SeriousDlqin2yrs']).collect()
  y_score = [float(i[0][1]) for i in results]
  y_true = [float(i[1]) for i in results]

  precision, recall, _ = precision_recall_curve(y_true, y_score, pos_label = 1)

  fig, ax = plt.subplots()
  ax.step(recall, precision)
  ax.set_xlim([0.0, 1.0])
  ax.set_ylim([0.0, 1.0])
  ax.set_xlabel('Recall')
  ax.set_ylabel('Precision')
  ax.set_title('Precision - Recall Curve')
  display(fig)

plotPRCurve(gbt_predictions)

# COMMAND ----------

tn, fp, fn, tp = gbt_confusion_matrix.ravel()
print(f"Sensitivity = TP / (TP + FN) = {tp/(tp+fn)}")
print(f"Specificity = TN / (TN + FP) = {tn/(tn+fp)}")

# COMMAND ----------

plotROCCurve(gbt_predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Selecting a better threshold for class separation

# COMMAND ----------

plotROCCurve(gbt_predictions, show_thresholds = True)

# COMMAND ----------

# select a different threshold for class separation, make predictions based on that threshold, and recalculate Precision, Recall, Sensitivity and Specificity.
from pyspark.sql.types import FloatType

get_positive_probability=udf(lambda v:float(v[1]),FloatType())


selected_threshold = 0.11
pred_colname = f'prediction-threshold'
gbt_predictions_threshold = gbt_predictions.withColumn(pred_colname, 
                                       F.when(get_positive_probability('probability') < selected_threshold,0)
                                                       .otherwise(1))
                                                       
display(gbt_predictions_threshold.select("probability", "prediction", pred_colname, "SeriousDlqin2yrs"))                                                

# COMMAND ----------

gbt_threshold_confusion_matrix = confusion_matrix(gbt_predictions_threshold.select("SeriousDlqin2yrs").collect(), gbt_predictions_threshold.select("prediction-threshold").collect())
plotConfusionMatrix(gbt_threshold_confusion_matrix)

# COMMAND ----------

tn, fp, fn, tp = gbt_threshold_confusion_matrix.ravel()

print(f"Precision = TP / (TP + FP) = {tp/(tp+fp)}")
print(f"Recall = TP / (TP + FN) = {tp/(tp+fn)}")

print(f"Sensitivity = TP / (TP + FN) = {tp/(tp+fn)}")
print(f"Specificity = TN / (TN + FP) = {tn/(tn+fp)}")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Hyperparameter Tuning

# COMMAND ----------

print(gbt_classifier.explainParams())

# COMMAND ----------

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator


paramGrid = (ParamGridBuilder()
             .addGrid(gbt_classifier.maxDepth, [5, 8])
             .addGrid(gbt_classifier.maxIter, [25, 40])
             .addGrid(gbt_classifier.stepSize, [0.1, 0.2])
             .build())


evaluator = BinaryClassificationEvaluator(
  rawPredictionCol="prediction", labelCol="SeriousDlqin2yrs", metricName="areaUnderROC")

cv = CrossValidator(estimator=gbt_pipeline, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)

# Train model. 
gbt_models_cv = cv.fit(trainingSDF)

# COMMAND ----------

best_model = gbt_models_cv.bestModel.stages[3]
print(best_model.explainParams())

# COMMAND ----------

# Make predictions.
gbt_cv_predictions = gbt_models_cv.transform(testingSDF)

# Select example rows to display.
display(gbt_cv_predictions.select("probability", "prediction", "SeriousDlqin2yrs"))

# COMMAND ----------

plotROCCurve(gbt_cv_predictions, show_thresholds = True)

# COMMAND ----------

selected_threshold = 0.11
pred_colname = f'prediction-threshold'
gbt_cv_predictions_threshold = gbt_cv_predictions.withColumn(pred_colname, 
                                       F.when(get_positive_probability('probability') < selected_threshold,0)
                                                       .otherwise(1))
                                                       
display(gbt_cv_predictions_threshold.select("probability", "prediction", pred_colname, "SeriousDlqin2yrs"))  

# COMMAND ----------

gbt_cv_threshold_confusion_matrix = confusion_matrix(gbt_cv_predictions_threshold.select("SeriousDlqin2yrs").collect(), gbt_cv_predictions_threshold.select("prediction-threshold").collect())
plotConfusionMatrix(gbt_cv_threshold_confusion_matrix)

# COMMAND ----------

tn, fp, fn, tp = gbt_cv_threshold_confusion_matrix.ravel()

print(f"Precision = TP / (TP + FP) = {tp/(tp+fp)}")
print(f"Recall = TP / (TP + FN) = {tp/(tp+fn)}")

print(f"Sensitivity = TP / (TP + FN) = {tp/(tp+fn)}")
print(f"Specificity = TN / (TN + FP) = {tn/(tn+fp)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Using Automated ML from Azure ML Service
# MAGIC 
# MAGIC We are now going to use the [AutoML feature](https://docs.microsoft.com/en-us/azure/machine-learning/service/concept-automated-ml) from the Azure Machine Learning Service SDK.
# MAGIC 
# MAGIC Automated machine learning, also referred to as automated ML, is the process of automating the time consuming, iterative tasks of machine learning model development. It allows data scientists, analysts, and developers to build ML models with high scale, efficiency, and productivity all while sustaining model quality. Automated ML is based on a breakthrough from our Microsoft Research division.
# MAGIC 
# MAGIC Traditional machine learning model development is resource-intensive, requiring significant domain knowledge and time to produce and compare dozens of models. Apply automated ML when you want Azure Machine Learning to train and tune a model for you using the target metric you specify. The service then iterates through ML algorithms paired with feature selections, where each iteration produces a model with a training score. The higher the score, the better the model is considered to "fit" your data.
# MAGIC 
# MAGIC 
# MAGIC We will provide the cleansed training data to Azure ML which will test multiple types of algorithms in order to maximize a certain evaluation criteria we define. As per the [initial challenge from kaggle](https://www.kaggle.com/c/GiveMeSomeCredit), the criteria of choice is AUC (Area Under Curve).
# MAGIC 
# MAGIC The validation during training is done by using cross validation in 5 folds.
# MAGIC 
# MAGIC After we are done, the best trained model will be evaluated against a separated dataset (the test dataset) in order to understand real _performance_.
# MAGIC 
# MAGIC ### Training using AutoML
# MAGIC 
# MAGIC In order to get things going, we first initialize our Workspace...

# COMMAND ----------

subscription_id = "6787a35f-386b-4845-91d1-695f24e0924b" # the Azure subscription ID you are using
azureml_resource_group = "spark-ml-workshop-25" #you should be owner or contributor
azureml_workspace_name = "azureml-lab-25" #your Azure Machine Learning workspace name

import azureml.core

# Check core SDK version number - based on build number of preview/master.
print("Azure ML SDK version:", azureml.core.VERSION)

from azureml.core import Workspace

ws = Workspace(workspace_name = azureml_workspace_name,
               subscription_id = subscription_id,
               resource_group = azureml_resource_group)

# Persist the subscription id, resource group name, and workspace name in aml_config/config.json.
ws.write_config()

# COMMAND ----------

ws = Workspace.from_config()

print('Workspace name: ' + ws.name, 
      'Azure region: ' + ws.location, 
      'Subscription id: ' + ws.subscription_id, 
      'Resource group: ' + ws.resource_group, sep = '\n')

# COMMAND ----------

# MAGIC %md
# MAGIC And then we make sure we have all the important libraries in place.

# COMMAND ----------

import logging
import os
import random
import time

from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
import numpy as np
import pandas as pd

import azureml.core
from azureml.core.experiment import Experiment
from azureml.core.workspace import Workspace
from azureml.train.automl import AutoMLConfig
from azureml.train.automl.run import AutoMLRun

# COMMAND ----------

# MAGIC %md
# MAGIC We prepare the experiment properties which will be provided once we issue a training request.

# COMMAND ----------

# Get the last seven letters of the username which will be used to build up exp name
import re

regexStr = r'^([^@]+)@[^@]+$'
emailStr = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply("user")
matchobj = re.search(regexStr, emailStr)
if not matchobj is None:
    if len(matchobj.group(1)) > 10:
        notebook_username = matchobj.group(1)[-10:]
    else:
        notebook_username = matchobj.group(1)
        
    print(notebook_username)
else:
    print("Did not match")

# COMMAND ----------

# Choose a name for the experiment and specify the project folder.
experiment_base_name = 'automl-scoring-'
experiment_suffix_name = notebook_username.replace(".", "") + "-" + str(random.randint(1000, 9999))

experiment_name = experiment_base_name + experiment_suffix_name
project_folder = './globalainight_projects/automl-credit-scring'

print(experiment_name)

experiment = Experiment(ws, experiment_name)

output = {}
output['SDK version'] = azureml.core.VERSION
output['Subscription ID'] = ws.subscription_id
output['Workspace Name'] = ws.name
output['Resource Group'] = ws.resource_group
output['Location'] = ws.location
output['Project Directory'] = project_folder
output['Experiment Name'] = experiment.name
pd.set_option('display.max_colwidth', -1)
pd.DataFrame(data = output, index = ['']).T

# COMMAND ----------

# MAGIC %md
# MAGIC Enabling diagnostics to understand better what's going on.

# COMMAND ----------

from azureml.telemetry import set_diagnostics_collection
set_diagnostics_collection(send_diagnostics = True)

# COMMAND ----------

# MAGIC %md
# MAGIC We save the cleansed training data to CSV files in the DBFS and then we load it separately in two dataflows:
# MAGIC - **X_train**: contains **_only_** the training variables
# MAGIC - **Y_train**: contains **_only_** the result

# COMMAND ----------

# prepare the data for AutoML
import azureml.dataprep as dataprep

training_sdf = trainingSDF
training_sdf = training_sdf.drop("Idx", "initialDebt")

training_sdf \
.drop("SeriousDlqin2yrs") \
.toPandas() \
.to_csv("/dbfs/FileStore/tables/constant-scoring-training-vars.csv")

training_sdf \
.select("SeriousDlqin2yrs") \
.toPandas() \
.to_csv("/dbfs/FileStore/tables/constant-scoring-training-res.csv")

X_train = dataprep.read_csv(path = "/dbfs/FileStore/tables/constant-scoring-training-vars.csv", separator = ',')
X_train = X_train.drop_columns("Column1")

Y_train = dataprep.read_csv(path = "/dbfs/FileStore/tables/constant-scoring-training-res.csv", separator = ',')
Y_train = Y_train.drop_columns("Column1")

# COMMAND ----------

# MAGIC %md
# MAGIC Checking to make sure we have data inside.

# COMMAND ----------

X_train.head(5)

# COMMAND ----------

Y_train.head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC AutoML in Azure may be configured by passing multiple properties through the [AutoML Config](https://docs.microsoft.com/en-us/python/api/azureml-train-automl/azureml.train.automl.automlconfig?view=azure-ml-py).
# MAGIC 
# MAGIC We are interested in the following:
# MAGIC 
# MAGIC |Property|Description|
# MAGIC |-|-|
# MAGIC |**task**|classification or regression|
# MAGIC |**primary_metric**|This is the metric that you want to optimize. Classification supports the following primary metrics: <br><i>accuracy</i><br><i>AUC_weighted</i><br><i>average_precision_score_weighted</i><br><i>norm_macro_recall</i><br><i>precision_score_weighted</i>|
# MAGIC |**primary_metric**|This is the metric that you want to optimize. Regression supports the following primary metrics: <br><i>spearman_correlation</i><br><i>normalized_root_mean_squared_error</i><br><i>r2_score</i><br><i>normalized_mean_absolute_error</i>|
# MAGIC |**iteration_timeout_minutes**|Time limit in minutes for each iteration.|
# MAGIC |**iterations**|Number of iterations. In each iteration AutoML trains a specific pipeline with the data.|
# MAGIC |**n_cross_validations**|Number of cross validation splits.|
# MAGIC |**spark_context**|Spark Context object. for Databricks, use spark_context=sc|
# MAGIC |**max_concurrent_iterations**|Maximum number of iterations to execute in parallel. This should be <= number of worker nodes in your Azure Databricks cluster.|
# MAGIC |**X**|(sparse) array-like, shape = [n_samples, n_features]|
# MAGIC |**y**|(sparse) array-like, shape = [n_samples, ], [n_samples, n_classes]<br>Multi-class targets. An indicator matrix turns on multilabel classification. This should be an array of integers.|
# MAGIC |**path**|Relative path to the project folder. AutoML stores configuration files for the experiment under this folder. You can specify a new empty folder.|
# MAGIC |**preprocess**|set this to True to enable pre-processing of data eg. string to numeric using one-hot encoding|
# MAGIC |**exit_score**|Target score for experiment. It is associated with the metric. eg. exit_score=0.995 will exit experiment after that|

# COMMAND ----------

automl_config = AutoMLConfig(task = 'classification',
                             debug_log = 'automl_errors.log',
                             primary_metric = 'AUC_weighted',
                             iteration_timeout_minutes = 5,
                             iterations = 10,
                             n_cross_validations = 5,
                             max_concurrent_iterations = 1, 
                             verbosity = logging.INFO,
                             spark_context=sc, #databricks/spark related
                             X = X_train, 
                             y = Y_train,
                             path = project_folder,
                             preprocess = True,
                             enable_voting_ensemble = False,
                             enable_stack_ensemble = False)

# COMMAND ----------

# MAGIC %md
# MAGIC We are now ready to submit a new run for our experiment.

# COMMAND ----------

local_run = experiment.submit(automl_config, show_output = True) # for higher runs please use show_output=False and use the below

# COMMAND ----------

# MAGIC %md
# MAGIC And once the run is finished, we are able to retrieve the best model as per the metric we have configured.

# COMMAND ----------

best_run, fitted_model = local_run.get_output()
print(best_run)
print(fitted_model)

# COMMAND ----------

print(fitted_model.steps)

# COMMAND ----------

# MAGIC %md 
# MAGIC **Portal URL for Monitoring Runs**
# MAGIC 
# MAGIC The following will provide a link to the web interface to explore individual run details and status.

# COMMAND ----------

displayHTML("<a href={} target='_blank'>Your experiment in Azure Portal: {}</a>".format(local_run.get_portal_url(), local_run.id))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluating the best model from AutoML
# MAGIC 
# MAGIC Now that we are done with training, we can move forward in evaluating how well this model will actually do on the test data.

# COMMAND ----------

automl_X_test = testingSDF.drop("Idx", "initialDebt","SeriousDlqin2yrs")
automl_Y_test = testingSDF.select("SeriousDlqin2yrs")

# COMMAND ----------

automl_predictions_pd = fitted_model.predict_proba(automl_X_test.toPandas())
tempdf = pd.concat([pd.DataFrame(automl_predictions_pd), automl_Y_test.toPandas()], axis=1)
automl_predictions = spark.createDataFrame(tempdf)
display(automl_predictions)

# COMMAND ----------

def plotROCCurve2(predictions, show_thresholds=False):
  results = predictions.select(['1', 'SeriousDlqin2yrs']).collect()
  y_score = [float(i[0]) for i in results]
  y_true = [float(i[1]) for i in results]

  fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label = 1)
  roc_auc = auc(fpr, tpr)

  fig, ax = plt.subplots()
  ax.plot(fpr, tpr, label='ROC curve (area = %0.4f)' % roc_auc)
  ax.plot([0, 1], [0, 1], 'k--')
  if show_thresholds:
      tr_idx = np.arange(385, len(thresholds), 700)
      for i in tr_idx:
        ax.plot(fpr[i], tpr[i], "xr")
        ax.annotate(xy=(fpr[i], tpr[i]), s="%0.3f" % thresholds[i])
  ax.set_xlim([0.0, 1.0])
  ax.set_ylim([0.0, 1.0])
  ax.set_xlabel('False Positive Rate (1 - Specificity)')
  ax.set_ylabel('True Positive Rate (Sensitivity)')
  ax.set_title('Receiver operating characteristic')
  ax.legend(loc="lower right")
  display(fig)

plotROCCurve2(automl_predictions, show_thresholds = True)

# COMMAND ----------

selected_threshold = 0.12
pred_colname = f'prediction-threshold'
automl_predictions_threshold = automl_predictions_SDF.withColumn(pred_colname, 
                                       F.when(F.col('1') < selected_threshold, 0)
                                                       .otherwise(1))
display(automl_predictions_threshold)

# COMMAND ----------

automl_threshold_confusion_matrix = confusion_matrix(automl_predictions_threshold.select("SeriousDlqin2yrs").collect(), automl_predictions_threshold.select("prediction-threshold").collect())
plotConfusionMatrix(automl_threshold_confusion_matrix)

# COMMAND ----------

tn, fp, fn, tp = automl_threshold_confusion_matrix.ravel()

print(f"Precision = TP / (TP + FP) = {tp/(tp+fp)}")
print(f"Recall = TP / (TP + FN) = {tp/(tp+fn)}")

print(f"Sensitivity = TP / (TP + FN) = {tp/(tp+fn)}")
print(f"Specificity = TN / (TN + FP) = {tn/(tn+fp)}")

# COMMAND ----------


