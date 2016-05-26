package MLlib

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel, SparseLogisticRegressionWithLBFGS}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkContext, SparkConf}

/**
 * Created by yuhao on 11/19/15.
 */
object LogisticRegressionTest {

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName(s"LogisticRegressionTest with $args").setMaster("local")
    val sc = new SparkContext(conf)

    Logger.getRootLogger.setLevel(Level.WARN)
    val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt").map(
      l => LabeledPoint(l.label, l.features.toSparse))

    // Split data into training (60%) and test (40%).
    val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)

    // Run training algorithm to build the model
    val model = new SparseLogisticRegressionWithLBFGS()
      .setNumClasses(2)
      .run(training)

    // Compute raw scores on the test set.
    val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }

    // Get evaluation metrics.
    val metrics = new MulticlassMetrics(predictionAndLabels)

    val precision = metrics.precision
    println("Precision = " + precision)


  }

}
