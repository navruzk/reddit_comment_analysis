# Databricks notebook source
import json
from pyspark.sql.types import ArrayType, StringType

## for user defined functions
## for tokenizing and removing stopwords
from pyspark.sql.functions import udf
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import Tokenizer

# COMMAND ----------

## create rdd to further work on it
rdd = sc.textFile("dbfs:/tmp/RC_2011-09.txt")

# COMMAND ----------

## txt file shows that it is txt of json
## keep only unix time information and comments
## also data that has 'deleted' info has been removed

## rdd keys are comment ids give under "name"
rdd_subset = rdd.map(
    lambda line: (
        json.loads(line)["name"],
        json.loads(line)["author"],
        json.loads(line)["author_flair_text"],
        json.loads(line)["created_utc"],
        json.loads(line)["parent_id"],
        json.loads(line)["ups"],
        json.loads(line)["downs"],
        json.loads(line)["retrieved_on"],
        json.loads(line)["subreddit"],
        json.loads(line)["body"],
    )
).filter(lambda line: line if "[deleted]" not in line[9] else None)

# COMMAND ----------

print(rdd_subset.take(1))

# COMMAND ----------


# COMMAND ----------

## here I create a dataframe from rdd and give column names
df = spark.createDataFrame(rdd_subset).toDF(
    "name",
    "author",
    "author_flair_text",
    "unix_time",
    "parent_id",
    "ups",
    "downs",
    "retrieved_on",
    "subreddit",
    "comment",
)

# COMMAND ----------

tokenizer = Tokenizer(inputCol="comment", outputCol="comment_tokens")
df_tokened = tokenizer.transform(df)

# COMMAND ----------

df_tokened.show(10)

# COMMAND ----------


def remove_int(x):
    return [a for a in x if a.isdigit() is False]


def short_words(x):
    return [a for a in x if len(a) <= 14]


remove_int_udf = udf(lambda line: remove_int(line), ArrayType(StringType()))
shorten_words_udf = udf(lambda line: short_words(line), ArrayType(StringType()))

df_tokened = df_tokened.withColumn(
    "comment_tokens_cleaned", remove_int_udf("comment_tokens")
)
df_tokened = df_tokened.withColumn(
    "comment_tokens_cleaned", shorten_words_udf("comment_tokens_cleaned")
)

# COMMAND ----------

stopword_remover = StopWordsRemover(
    inputCol="comment_tokens_cleaned", outputCol="comment_cleaned"
)
df_cleaned = stopword_remover.transform(df_tokened)
