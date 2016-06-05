package org.apache.spark.mllib.sparselr

import org.apache.spark.mllib.sparselr.Utils._

abstract class Gradient extends Serializable {
  /**
   * Compute the gradient and loss given the features of a single data point.
   *
   * @param data features for one data point
   * @param label label for this data point
   * @param weights weights/coefficients corresponding to features
   *
   * @return (gradient: Vector, loss: Double)
   */
  def compute(
              data: Vector,
              label: Double,
              weights: Vector): (Vector, Double)

  /**
   * Compute the gradient and loss given the features of a single data point.
   *
   * @param data features for one data point
   * @param label label for this data point
   * @param weights weights/coefficients corresponding to features
   *
   * @return loss: Double
   */
  def compute(
               data: Vector,
               label: Double,
               weights: Vector,
               cumGradient: Vector): Double
}

class LogisticGradient extends Gradient {
  override def compute(
                data: Vector,
                label: Double,
                weights: Vector): (Vector, Double) = {

    val gradient = org.apache.spark.mllib.sparselr.Utils.Vectors.hashSparseVector()
    val loss = compute(data, label, weights, gradient)
    (gradient, loss)
  }

  override def compute(
                data: Vector,
                label: Double,
                weights: Vector,
                cumGradient: Vector): Double = {
        val margin = -1.0 * BLAS.dot(data, weights)
        val multiplier = (1.0 / (1.0 + math.exp(margin))) - label

        BLAS.axpy(multiplier, data, cumGradient)

        if (label > 0) {
          // The following is equivalent to log(1 + exp(margin)) but more numerically stable.
          LRUtils.log1pExp(margin)
        } else {
          LRUtils.log1pExp(margin) - margin
        }
  }
}
