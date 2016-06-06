package org.apache.spark.mllib.sparselr

import org.apache.spark.mllib.sparselr.Utils.Vector
import org.apache.spark.rdd.RDD

trait Optimizer extends Serializable{
  def optimize(
           data: RDD[(Double, Vector)],
           initialWeights: Vector): Vector
}
