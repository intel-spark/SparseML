import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.clustering.{KMeans, SparseKMeans}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.{SparseVector, Vectors, Vector}

import scala.util.Random

/**
 * Created by yuhao on 1/23/16.
 */
object KMeanTest {

  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val conf = new SparkConf().setAppName(s"kmeans: ${args.mkString(",")}").setMaster("local[4]")
    val sc = new SparkContext(conf)

    val k = args(0).toInt
    val dimension = args(1).toInt
    val recordNum = args(2).toInt
    val sparsity = args(3).toDouble
    val iterations = args(4).toInt
    val means = args(5)

    val data: RDD[Vector] = sc.parallelize(1 to recordNum).map(i => {
      val ran = new Random()
      val indexArr = (1 to (dimension * sparsity).toInt).map(in => ran.nextInt(dimension)).sorted.toArray
      val valueArr = (1 to (dimension * sparsity).toInt).map(in => ran.nextDouble()).sorted.toArray
      val vec: Vector = new SparseVector(dimension, indexArr, valueArr)
      vec
    }).cache()
    println(data.count() + " records generated")

    val st = System.nanoTime()


    if(means == "my") {
      println("running sparse kmeans")
      val model = new SparseKMeans()
        .setK(k)
        .setInitializationMode("random")
        .setMaxIterations(iterations)
        .run(data)

      println((System.nanoTime() - st) / 1e9 + " seconds cost")
      println("final clusters:")
      println(model.clusterCenters.map(v => v.numNonzeros).mkString("\n"))
    } else {
      println("running mllib kmeans")
      val model = new KMeans()
        .setK(k)
        .setInitializationMode("random")
        .setMaxIterations(iterations)
        .run(data)

      println((System.nanoTime() - st) / 1e9 + " seconds cost")
      println("final clusters:")
      println(model.clusterCenters.map(v => v.numNonzeros).mkString("\n"))
    }

    sc.stop()
  }

}
