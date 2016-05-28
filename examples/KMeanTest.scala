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
    val conf = new SparkConf().setAppName(s"kmeans: ${args.mkString(",")}")
    val sc = new SparkContext(conf)
    val k = 10

    val dimension = 10000
    val recordNum = 100000
    val sparsity = 0.001
    val iterations = 10

    val data: RDD[Vector] = sc.parallelize(1 to recordNum).map(i => {
      val ran = new Random()
      val indexArr = (1 to (dimension * sparsity).toInt).map(in => ran.nextInt(dimension)).sorted.toArray
      val valueArr = (1 to (dimension * sparsity).toInt).map(in => ran.nextDouble()).sorted.toArray
      val vec: Vector = new SparseVector(dimension, indexArr, valueArr)
      vec
    }).cache()
    println(data.count() + " records generated")

    val st = System.nanoTime()

    val model = new KMeans()
      .setK(k)
      .setInitializationMode("k-means||")
      .setMaxIterations(iterations)
      .run(data)

    println((System.nanoTime() - st) / 1e9 + " seconds cost")
    println("final clusters:")
    println(model.clusterCenters.map(v => v.numNonzeros).mkString("\n"))


    sc.stop()
  }

}
