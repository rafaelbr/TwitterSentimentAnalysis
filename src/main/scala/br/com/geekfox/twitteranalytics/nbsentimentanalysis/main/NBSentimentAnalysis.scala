package br.com.geekfox.twitteranalytics.nbsentimentanalysis.main

import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.ml.feature.{CountVectorizer, Tokenizer}
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.{Pipeline, classification}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator


object NBSentimentAnalysis {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().appName("TwitterSentimentAnalysis").master("local[*]").getOrCreate()

    import spark.implicits._

    val tweets_df = spark.read.format("csv").load("src/main/resources/training.1600000.processed.noemoticon.csv")

    val df = tweets_df
      .selectExpr("CAST(_c0 AS INT)", "_c5")
      .map(row => (if (row.getInt(0) == 4) 1 else 0, row.getString(1)))
      .selectExpr("_1 AS label", "_2 AS tweet")



    df.show(10)


    val tokenizer = new Tokenizer().setInputCol("tweet").setOutputCol("words")

    val vectorizer = new CountVectorizer().setInputCol("words").setOutputCol("features")

    val estimator = new NaiveBayes().setFeaturesCol("features").setLabelCol("label")

    val pipeline = new Pipeline().setStages(Array(tokenizer, vectorizer, estimator))

    val nbModel = pipeline.fit(df)

    nbModel.write.overwrite().save("src/main/resources/model")

    val test_df = spark
      .read
      .format("csv")
      .load("src/main/resources/testdata.manual.2009.06.14.csv")
      .selectExpr("CAST(_c0 AS INT)", "_c5")
      .where("_c0 != 2")
      .map(row => (if (row.getInt(0) == 4) 1 else 0, row.getString(1)))
      .selectExpr("_1 AS label", "_2 AS tweet")



    val result_df = nbModel.transform(test_df)

    result_df.show(10)

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(result_df)

    println("Accuracy " + accuracy)

    spark.stop()

  }
}
