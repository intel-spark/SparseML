package org.apache.spark.mllib.sparselr

import org.apache.spark.mllib.sparselr.Utils._
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.annotation.{Experimental, DeveloperApi}
import org.apache.spark.{SparkEnv, Logging}
import org.apache.spark.rdd.RDD

/**
 * Class used to solve an optimization problem using Gradient Descent.
 * @param gradient Gradient function to be used.
 * @param updater Updater to be used to update weights after every iteration.
 */
@DeveloperApi
class GradientDescent (private var gradient: Gradient, private var updater: Updater)
  extends Optimizer with Logging {

  private var stepSize: Double = 1.0
  private var numIterations: Int = 100
  private var regParam: Double = 0.0
  private var miniBatchFraction: Double = 1.0

  /**
   * Set the initial step size of SGD for the first step. Default 1.0.
   * In subsequent steps, the step size will decrease with stepSize/sqrt(t)
   */
  def setStepSize(step: Double): this.type = {
    this.stepSize = step
    this
  }

  /**
   * :: Experimental ::
   * Set fraction of data to be used for each SGD iteration.
   * Default 1.0 (corresponding to deterministic/classical gradient descent)
   */
  @Experimental
  def setMiniBatchFraction(fraction: Double): this.type = {
    this.miniBatchFraction = fraction
    this
  }

  /**
   * Set the number of iterations for SGD. Default 100.
   */
  def setNumIterations(iters: Int): this.type = {
    this.numIterations = iters
    this
  }

  /**
   * Set the regularization parameter. Default 0.0.
   */
  def setRegParam(regParam: Double): this.type = {
    this.regParam = regParam
    this
  }

  /**
   * Set the gradient function (of the loss function of one single data example)
   * to be used for SGD.
   */
  def setGradient(gradient: Gradient): this.type = {
    this.gradient = gradient
    this
  }

  /**
   * Set the updater function to actually perform a gradient step in a given direction.
   * The updater is responsible to perform the update from the regularization term as well,
   * and therefore determines what kind or regularization is used, if any.
   */
  def setUpdater(updater: Updater): this.type = {
    this.updater = updater
    this
  }

  /**
   * :: DeveloperApi ::
   * Runs gradient descent on the given training data.
   * @param data training data
   * @param initialWeights initial weights
   * @return solution vector
   */
  @DeveloperApi
  def optimize(
               data: RDD[(Double, Vector)],
               initialWeights: Vector): Vector = {
    val (weights, _) = GradientDescent.runMiniBatchSGD(
      data,
      gradient,
      updater,
      stepSize,
      numIterations,
      regParam,
      miniBatchFraction,
      initialWeights)
    weights
  }
}

/**
 * :: DeveloperApi ::
 * Top-level method to run gradient descent.
 */
@DeveloperApi
object GradientDescent extends Logging {
    /**
     * Run stochastic gradient descent (SGD) in parallel using mini batches.
     * In each iteration, we sample a subset (fraction miniBatchFraction) of the total data
     * in order to compute a gradient estimate.
     * Sampling, and averaging the subgradients over this subset is performed using one standard
     * spark map-reduce in each iteration.
     *
     * @param data - Input data for SGD. RDD of the set of data examples, each of
     *             the form (label, [feature values]).
     * @param gradient - Gradient object (used to compute the gradient of the loss function of
     *                 one single data example)
     * @param updater - Updater function to actually perform a gradient step in a given direction.
     * @param numIterations - number of iterations that SGD should be run.
     * @param miniBatchFraction - fraction of the input data set that should be used for
     *                          one iteration of SGD. Default value 1.0.
     *
     * @return A tuple containing two elements. The first element is a column matrix containing
     *         weights for every feature, and the second element is an array containing the
     *         stochastic loss computed for every iteration.
     */
    def runMiniBatchSGD(
         data: RDD[(Double, Vector)],
         gradient: Gradient,
         updater: Updater,
         stepSize: Double,
         numIterations: Int,
         regParam: Double,
         miniBatchFraction: Double,
         initialWeights: Vector): (Vector, Array[Double]) = {
      val stochasticLossHistory = new ArrayBuffer[Double](numIterations)

      val numExamples = data.count()
      // if no data, return initial weights to avoid NaNs
      if (numExamples == 0) {
        logWarning("GradientDescent.runMiniBatchSGD returning initial weights, no data found")
        return (initialWeights, stochasticLossHistory.toArray)
      }

      if (numExamples * miniBatchFraction < 1) {
        logWarning("The miniBatchFraction is too small")
      }

      // Initialize weights as a column vector
      var weights = Vectors.dense(initialWeights.toArray)

      var regVal = updater.compute(
        weights, new HashedSparseVector(), 0, 1, regParam)._2

      def simulateWeights(first: (HashedSparseVector, Double, Int), second: (HashedSparseVector, Double, Int))
      : (HashedSparseVector, Double, Int) = {
        val iterSecond = second._1.hashmap.int2DoubleEntrySet.fastIterator()
        while (iterSecond.hasNext()) {
          val entry = iterSecond.next()
          first._1.hashmap.addTo(entry.getIntKey, entry.getDoubleValue)
        }
        (first._1, first._2 + second._2, first._3 + second._3)
      }

      for (i <- 1 to numIterations) {
        val bcWeights = data.context.broadcast(weights)
        // Sample a subset (fraction miniBatchFraction) of the total data
        // compute and sum up the subgradients on this subset (this is one map-reduce)
        val (gradientSum, lossSum, miniBatchSize) = data.sample(false, miniBatchFraction, 42 + i)
//        val (gradientSum, lossSum, miniBatchSize) = data
          .mapPartitions { points =>
          var loss = 0.0
          val gradientPerPartition = new HashedSparseVector()
          var size = 0
          points.foreach { point =>
            loss += gradient.compute(point._2, point. _1, bcWeights.value, gradientPerPartition)
            size += 1
          }
          Iterator((gradientPerPartition, loss, size))
        }.reduce(simulateWeights)

        if (miniBatchSize > 0) {
          stochasticLossHistory.append(lossSum / miniBatchSize)

          BLAS.dot(1 / miniBatchSize.toDouble, gradientSum)
          val update = updater.compute(weights, gradientSum, stepSize, i, regParam)
          weights = update._1
          regVal = update._2
        } else {
          logWarning(s"Iteration ($i/$numIterations). The size of sampled batch is zero")
        }
        SparkEnv.get.blockManager.removeBroadcast(bcWeights.id, true)
      }

      logInfo("GradientDescent.runMiniBatchSGD finished. Last 10 stochastic losses %s".format(
        stochasticLossHistory.takeRight(10).mkString(", ")))
      (weights, stochasticLossHistory.toArray)
    }
}
