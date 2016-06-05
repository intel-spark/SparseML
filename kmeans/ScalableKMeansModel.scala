/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.spark.mllib.clustering

import org.apache.spark.annotation.Since
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.pmml.PMMLExportable
import org.apache.spark.rdd.RDD

/**
 * A clustering model for K-means. Each point belongs to the cluster with the closest center.
 */
@Since("0.8.0")
class ScalableKMeansModel @Since("1.1.0") (@Since("1.0.0")override val clusterCenters: Array[Vector])
  extends KMeansModel(clusterCenters) with Serializable with PMMLExportable {

  /**
   * Returns the cluster index that a given point belongs to.
   */
  @Since("0.8.0")
  override def predict(point: Vector): Int = {
    ScalableKMeans.findClosest(clusterCentersWithNorm, new VectorWithNorm(point))._1
  }

  /**
   * Maps given points to their cluster indices.
   */
  @Since("1.0.0")
  override def predict(points: RDD[Vector]): RDD[Int] = {
    val centersWithNorm = clusterCentersWithNorm
    val bcCentersWithNorm = points.context.broadcast(centersWithNorm)
    points.map(p => ScalableKMeans.findClosest(bcCentersWithNorm.value, new VectorWithNorm(p))._1)
  }


  /**
   * Return the K-means cost (sum of squared distances of points to their nearest center) for this
   * model on the given data.
   */
  @Since("0.8.0")
  override def computeCost(data: RDD[Vector]): Double = {
    val centersWithNorm = clusterCentersWithNorm
    val bcCentersWithNorm = data.context.broadcast(centersWithNorm)
    data.map(p => ScalableKMeans.pointCost(bcCentersWithNorm.value, new VectorWithNorm(p))).sum()
  }

  private def clusterCentersWithNorm: Iterable[VectorWithNorm] =
    clusterCenters.map(new VectorWithNorm(_))

}
