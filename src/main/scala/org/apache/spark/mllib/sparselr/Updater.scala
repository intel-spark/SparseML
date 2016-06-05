package org.apache.spark.mllib.sparselr

import scala.math._
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.mllib.sparselr.Utils._
import breeze.linalg.{DenseVector => BDV, norm}
/**
 * :: DeveloperApi ::
 * Class used to perform renorm.
 *
 */
@DeveloperApi
abstract class Updater extends Serializable {
  /**
   * Compute an updated value for weights given the gradient, stepSize, iteration number and
   * regularization parameter. Also returns the regularization value regParam * R(w)
   * computed using the *updated* weights.
   *
   * @param weightsOld - Column matrix of size dx1 where d is the number of features.
   * @param gradient - Column matrix of size dx1 where d is the number of features.
   * @param stepSize - step size across iterations
   * @param iter - Iteration number
   * @param regParam - Regularization parameter
   *
   * @return A tuple of 2 elements. The first element is a column matrix containing updated weights,
   *         and the second element is the regularization value computed using updated weights.
   */
  def compute(
              weightsOld: Vector,
              gradient: Vector,
              stepSize: Double,
              iter: Int,
              regParam: Double): (Vector, Double)
}

/**
 * :: DeveloperApi ::
 * A simple updater for gradient descent *without* any regularization.
 * Uses a step-size decreasing with the square root of the number of iterations.
 */
@DeveloperApi
class SimpleUpdater extends Updater {
  override def compute(
                        weightsOld: Vector,
                        gradient: Vector,
                        stepSize: Double,
                        iter: Int,
                        regParam: Double): (Vector, Double) = {
    val thisIterStepSize = stepSize / math.sqrt(iter)
    val weights = weightsOld.copy
    BLAS.axpy(-thisIterStepSize, gradient, weights)
    (weights, 0)
  }
}

/**
 * :: DeveloperApi ::
 * Updater for L1 regularized problems.
 *          R(w) = ||w||_1
 * Uses a step-size decreasing with the square root of the number of iterations.

 * Instead of subgradient of the regularizer, the proximal operator for the
 * L1 regularization is applied after the gradient step. This is known to
 * result in better sparsity of the intermediate solution.
 *
 * The corresponding proximal operator for the L1 norm is the soft-thresholding
 * function. That is, each weight component is shrunk towards 0 by shrinkageVal.
 *
 * If w >  shrinkageVal, set weight component to w-shrinkageVal.
 * If w < -shrinkageVal, set weight component to w+shrinkageVal.
 * If -shrinkageVal < w < shrinkageVal, set weight component to 0.
 *
 * Equivalently, set weight component to signum(w) * max(0.0, abs(w) - shrinkageVal)
 */
@DeveloperApi
class L1Updater extends Updater {
  override def compute(
                        weightsOld: Vector,
                        gradient: Vector,
                        stepSize: Double,
                        iter: Int,
                        regParam: Double): (Vector, Double) = {
    val thisIterStepSize = stepSize / math.sqrt(iter)
    // Take gradient step
    val weights = weightsOld.copy
    BLAS.axpy(-thisIterStepSize, gradient, weights)
    // Apply proximal operator (soft thresholding)
    val shrinkageVal = regParam * thisIterStepSize
    var i = 0
    val len = weights.toArray.length
    while (i < len) {
      val wi = weights(i)
      weights(i) = signum(wi) * max(0.0, abs(wi) - shrinkageVal)
      i += 1
    }
    (weights, norm(new BDV[Double](weights.toArray).toDenseVector, 1.0) * regParam)
  }
}

/**
 * :: DeveloperApi ::
 * Updater for L2 regularized problems.
 *          R(w) = 1/2 ||w||^2
 * Uses a step-size decreasing with the square root of the number of iterations.
 */
@DeveloperApi
class SquaredL2Updater extends Updater {
  override def compute(
                        weightsOld: Vector,
                        gradient: Vector,
                        stepSize: Double,
                        iter: Int,
                        regParam: Double): (Vector, Double) = {
    // add up both updates from the gradient of the loss (= step) as well as
    // the gradient of the regularizer (= regParam * weightsOld)
    // w' = w - thisIterStepSize * (gradient + regParam * w)
    // w' = (1 - thisIterStepSize * regParam) * w - thisIterStepSize * gradient
    val thisIterStepSize = stepSize / math.sqrt(iter)
    val weights = weightsOld.copy
    for (i <- 0 until weights.toArray.length) {
      weights(i) *= (1.0 - thisIterStepSize * regParam)
    }
    BLAS.axpy(-thisIterStepSize, gradient, weights)
    val normValue = norm(new BDV[Double](weights.toArray).toDenseVector, 2.0)
    (weights, 0.5 * regParam * normValue * normValue)
  }
}