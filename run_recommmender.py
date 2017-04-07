from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql.types import *
from pyspark.sql.functions import from_unixtime

import sys
import math


def get_ratings_df(ratings_path, spark):
    schema = StructType([
        StructField('user', IntegerType(), True),
        StructField('movie', IntegerType(), True),
        StructField('rating', DoubleType(), True),
        StructField('timestamp', IntegerType(), True)
    ])

    ratings_df = spark.read\
        .csv(ratings_path, header=True, schema=schema)

    timestamp = from_unixtime(ratings_df.timestamp).cast("timestamp")
    ratings_df = ratings_df\
        .withColumn("timestamp", timestamp)

    return ratings_df


def create_model(train_df):
    model = ALS(userCol='user', itemCol='movie', ratingCol='rating')
    return model.fit(train_df)


def make_predictions(model, test_df):
    predictions = sorted(model
                         .transform(test_df)
                         .collect(),
                         key=lambda r: r[0])

    predictions = [row for row in predictions
                   if not math.isnan(row.prediction)]

    return predictions


def evaluate_predictions(predictions):
    def error(row):
        return (row.prediction - row.rating) ** 2

    return float(sum([error(row)
                 for row in predictions]))/len(predictions)


def main():
    ratings_path = sys.argv[1]

    spark = SparkSession.builder\
        .appName("movie-lens")\
        .getOrCreate()

    ratings_df = get_ratings_df(ratings_path, spark)
    train_df = ratings_df.where("timestamp < '2016-01-01 0:00:00'")
    test_df = ratings_df.where("timestamp >= '2016-01-01 0:00:00' \
        AND timestamp < '2016-02-01 0:00:00'")
    print('train_df count:{}'.format(train_df.count()))
    print('test_df count:{}'.format(test_df.count()))

    model = create_model(train_df)
    predictions = make_predictions(model, test_df)
    mse = evaluate_predictions(predictions)
    # we loose data, b.c. of newer movies that have come out that
    # are not found in the data set i.e. we'll have new users &
    # new movies that don't exist in the matrix
    print('MSE of {} for {} predictions'
          .format(mse, len(predictions)))


if __name__ == '__main__':
    main()
