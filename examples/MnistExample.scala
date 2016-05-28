import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.SVM
import org.apache.spark.mllib.clustering.SparseKMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.SparkSession

object MnistExample {


  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)
    val spark = SparkSession.builder.appName("svm").master("local[8]").getOrCreate()

    val trainRDD = spark.sparkContext.textFile("data/mnist/mnist_train.csv", 8)
      .map(line => line.split(",")).map(arr => arr.map(_.toDouble))
      .map(arr =>  Vectors.dense(arr.slice(1, 785)))

    val model = new SparseKMeans()
      .setK(10)
      .setInitializationMode("random")
      .setMaxIterations(100)
      .run(trainRDD)

    println("final clusters:")
    println(model.clusterCenters.map(v => v.numNonzeros).mkString("\n"))
  }

}
