package sparseLR

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, SparseLogisticRegressionWithLBFGS}
import org.apache.spark.mllib.linalg.{SparseVector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkContext, SparkConf}

import scala.util.Random

/**
 * Created by yuhao on 5/23/16.
 */
object SparseLRTest {

  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val conf = new SparkConf().setAppName(s"LogisticRegressionTest with $args").setMaster("local")
    val sc = new SparkContext(conf)

    val dimension = 100000
    val recordNum = 10000

    val data = sc.parallelize(1 to recordNum).map(i => {
      val ran = new Random()
      val indexArr = (1 to dimension/100).map(in => ran.nextInt(dimension)).sorted.toArray
      val valueArr = (1 to dimension/100).map(in => ran.nextDouble()).sorted.toArray
      val vec = new SparseVector(dimension, indexArr, valueArr)
      LabeledPoint(ran.nextInt(10).toDouble, vec)
    }).cache()
    println(data.count() + " records generated")

    val st = System.nanoTime()
    // Run training algorithm to build the model
    val model = new SparseLogisticRegressionWithLBFGS()
      .setNumClasses(10)
      .run(data)


    println((System.nanoTime() - st) / 1e9 + " seconds cost")


  }

}
