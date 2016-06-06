package org.apache.spark.mllib.sparselr

import it.unimi.dsi.fastutil.ints.Int2IntOpenHashMap
import org.apache.spark.mllib.sparselr.Utils._
import org.apache.spark.SparkEnv
import org.apache.spark.rdd.RDD
import org.apache.spark.broadcast.Broadcast

object LogisticRegression {
    def train(input: RDD[(Array[Double], Matrix)],
              optimizer: Optimizer
              ): (Array[Int], Array[Double]) = {

      val hdfsIndex2global = new Int2IntOpenHashMap()
      var index = 0

      input.map { point =>
        point._2 match {
          case x: CompressedSparseMatrix =>
            println("x.length" + x.mappings.length)
          case _ =>
            throw new IllegalArgumentException(s"dot doesn't support ${input.getClass}.")
        }
      }.count

      val global2hdfsIndex = input.map { point =>
        point._2 match {
          case x: CompressedSparseMatrix =>
            x.mappings
          case _ =>
            throw new IllegalArgumentException(s"dot doesn't support ${input.getClass}.")
        }
      }.collect().flatMap(t => t).distinct

      global2hdfsIndex.foreach{value =>
        hdfsIndex2global.put(value, index)
        index += 1
      }

      val bcHdfsIndex2global = input.context.broadcast(hdfsIndex2global)

      val examples = input.map(global2globalMapping(bcHdfsIndex2global)).cache()

      val numTraining = examples.count()
      println(s"Training: $numTraining.")

      SparkEnv.get.blockManager.removeBroadcast(bcHdfsIndex2global.id, true)

      val examplesTest = examples.mapPartitions(_.flatMap {
        case (y, part) => part.asInstanceOf[CompressedSparseMatrix].tupletIterator(y)})

      val weights = Vectors.dense(new Array[Double](global2hdfsIndex.size))

      val newWeights = optimizer.optimize(examplesTest, weights)

      ((global2hdfsIndex, newWeights.toArray))
    }

  //globalId to localId for mappings in Matrix
    def global2globalMapping(bchdfsIndex2global: Broadcast[Int2IntOpenHashMap])
                     (partition: (Array[Double], Matrix)): (Array[Double], Matrix) = {
      val hdfsIndex2global = bchdfsIndex2global.value

      partition._2 match {
        case x: CompressedSparseMatrix =>
          val local2hdfsIndex = x.mappings
          for (i <- 0 until local2hdfsIndex.length) {
            local2hdfsIndex(i) = hdfsIndex2global.get(local2hdfsIndex(i))
          }
        case _ =>
          throw new IllegalArgumentException(s"dot doesn't support ${partition.getClass}.")
      }
      partition
    }
}
