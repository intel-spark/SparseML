package org.apache.spark.mllib.optimization

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.mllib.linalg.{SparseVector, DenseVector, Vector, Vectors}
import org.apache.spark.mllib.linalg.BLAS.{axpy, dot, scal}
import org.apache.spark.mllib.util.MLUtils

/**
 * :: DeveloperApi ::
 * Compute gradient and loss for a multinomial logistic loss function, as used
 * in multi-class classification (it is also used in binary logistic regression).
 *
 * In `The Elements of Statistical Learning: Data Mining, Inference, and Prediction, 2nd Edition`
 * by Trevor Hastie, Robert Tibshirani, and Jerome Friedman, which can be downloaded from
 * http://statweb.stanford.edu/~tibs/ElemStatLearn/ , Eq. (4.17) on page 119 gives the formula of
 * multinomial logistic regression model. A simple calculation shows that
 *
 * {{{
 * P(y=0|x, w) = 1 / (1 + \sum_i^{K-1} \exp(x w_i))
 * P(y=1|x, w) = exp(x w_1) / (1 + \sum_i^{K-1} \exp(x w_i))
 *   ...
 * P(y=K-1|x, w) = exp(x w_{K-1}) / (1 + \sum_i^{K-1} \exp(x w_i))
 * }}}
 *
 * for K classes multiclass classification problem.
 *
 * The model weights w = (w_1, w_2, ..., w_{K-1})^T becomes a matrix which has dimension of
 * (K-1) * (N+1) if the intercepts are added. If the intercepts are not added, the dimension
 * will be (K-1) * N.
 *
 * As a result, the loss of objective function for a single instance of data can be written as
 * {{{
 * l(w, x) = -log P(y|x, w) = -\alpha(y) log P(y=0|x, w) - (1-\alpha(y)) log P(y|x, w)
 *         = log(1 + \sum_i^{K-1}\exp(x w_i)) - (1-\alpha(y)) x w_{y-1}
 *         = log(1 + \sum_i^{K-1}\exp(margins_i)) - (1-\alpha(y)) margins_{y-1}
 * }}}
 *
 * where \alpha(i) = 1 if i != 0, and
 *       \alpha(i) = 0 if i == 0,
 *       margins_i = x w_i.
 *
 * For optimization, we have to calculate the first derivative of the loss function, and
 * a simple calculation shows that
 *
 * {{{
 * \frac{\partial l(w, x)}{\partial w_{ij}}
 *   = (\exp(x w_i) / (1 + \sum_k^{K-1} \exp(x w_k)) - (1-\alpha(y)\delta_{y, i+1})) * x_j
 *   = multiplier_i * x_j
 * }}}
 *
 * where \delta_{i, j} = 1 if i == j,
 *       \delta_{i, j} = 0 if i != j, and
 *       multiplier =
 *         \exp(margins_i) / (1 + \sum_k^{K-1} \exp(margins_i)) - (1-\alpha(y)\delta_{y, i+1})
 *
 * If any of margins is larger than 709.78, the numerical computation of multiplier and loss
 * function will be suffered from arithmetic overflow. This issue occurs when there are outliers
 * in data which are far away from hyperplane, and this will cause the failing of training once
 * infinity / infinity is introduced. Note that this is only a concern when max(margins) > 0.
 *
 * Fortunately, when max(margins) = maxMargin > 0, the loss function and the multiplier can be
 * easily rewritten into the following equivalent numerically stable formula.
 *
 * {{{
 * l(w, x) = log(1 + \sum_i^{K-1}\exp(margins_i)) - (1-\alpha(y)) margins_{y-1}
 *         = log(\exp(-maxMargin) + \sum_i^{K-1}\exp(margins_i - maxMargin)) + maxMargin
 *           - (1-\alpha(y)) margins_{y-1}
 *         = log(1 + sum) + maxMargin - (1-\alpha(y)) margins_{y-1}
 * }}}
 *
 * where sum = \exp(-maxMargin) + \sum_i^{K-1}\exp(margins_i - maxMargin) - 1.
 *
 * Note that each term, (margins_i - maxMargin) in \exp is smaller than zero; as a result,
 * overflow will not happen with this formula.
 *
 * For multiplier, similar trick can be applied as the following,
 *
 * {{{
 * multiplier = \exp(margins_i) / (1 + \sum_k^{K-1} \exp(margins_i)) - (1-\alpha(y)\delta_{y, i+1})
 *            = \exp(margins_i - maxMargin) / (1 + sum) - (1-\alpha(y)\delta_{y, i+1})
 * }}}
 *
 * where each term in \exp is also smaller than zero, so overflow is not a concern.
 *
 * For the detailed mathematical derivation, see the reference at
 * http://www.slideshare.net/dbtsai/2014-0620-mlor-36132297
 *
 * @param numClasses the number of possible outcomes for k classes classification problem in
 *                   Multinomial Logistic Regression. By default, it is binary logistic regression
 *                   so numClasses will be set to 2.
 */
@DeveloperApi
class SparseLogisticGradient(numClasses: Int) extends Gradient {

  def this() = this(2)

  override def compute(
      data: Vector,
      label: Double,
      weights: Vector,
      cumGradient: Vector): Double = {
    val dataSize = data.size

    // (weights.size / dataSize + 1) is number of classes
    require(weights.size % dataSize == 0 && numClasses == weights.size / dataSize + 1)
    numClasses match {
      case 2 =>
        /**
         * For Binary Logistic Regression.
         *
         * Although the loss and gradient calculation for multinomial one is more generalized,
         * and multinomial one can also be used in binary case, we still implement a specialized
         * binary version for performance reason.
         */
        val margin = -1.0 * dot(data, weights)
        val multiplier = (1.0 / (1.0 + math.exp(margin))) - label
        axpy(multiplier, data, cumGradient)
        if (label > 0) {
          // The following is equivalent to log(1 + exp(margin)) but more numerically stable.
          MLUtils.log1pExp(margin)
        } else {
          MLUtils.log1pExp(margin) - margin
        }
      case _ =>
        /**
         * For Multinomial Logistic Regression.
         */
        val weightsArray = weights match {
          case dv: DenseVector => dv.values
          case sv: SparseVector => sv.toArray
          case _ =>
            throw new IllegalArgumentException(
              s"weights only supports dense vector but got type ${weights.getClass}.")
        }
        val cumGradientArray = cumGradient match {
          case dv: DenseVector => dv.values
          case sv: SparseVector => sv.toArray
          case _ =>
            throw new IllegalArgumentException(
              s"cumGradient only supports dense vector but got type ${cumGradient.getClass}.")
        }

        // marginY is margins(label - 1) in the formula.
        var marginY = 0.0
        var maxMargin = Double.NegativeInfinity
        var maxMarginIndex = 0

        val margins = Array.tabulate(numClasses - 1) { i =>
          var margin = 0.0
          data.foreachActive { (index, value) =>
            if (value != 0.0) margin += value * weightsArray((i * dataSize) + index)
          }
          if (i == label.toInt - 1) marginY = margin
          if (margin > maxMargin) {
            maxMargin = margin
            maxMarginIndex = i
          }
          margin
        }

        /**
         * When maxMargin > 0, the original formula will cause overflow as we discuss
         * in the previous comment.
         * We address this by subtracting maxMargin from all the margins, so it's guaranteed
         * that all of the new margins will be smaller than zero to prevent arithmetic overflow.
         */
        val sum = {
          var temp = 0.0
          if (maxMargin > 0) {
            for (i <- 0 until numClasses - 1) {
              margins(i) -= maxMargin
              if (i == maxMarginIndex) {
                temp += math.exp(-maxMargin)
              } else {
                temp += math.exp(margins(i))
              }
            }
          } else {
            for (i <- 0 until numClasses - 1) {
              temp += math.exp(margins(i))
            }
          }
          temp
        }

        for (i <- 0 until numClasses - 1) {
          val multiplier = math.exp(margins(i)) / (sum + 1.0) - {
            if (label != 0.0 && label == i + 1) 1.0 else 0.0
          }
          data.foreachActive { (index, value) =>
            if (value != 0.0) cumGradientArray(i * dataSize + index) += multiplier * value
          }
        }

        val loss = if (label > 0.0) math.log1p(sum) - marginY else math.log1p(sum)

        if (maxMargin > 0) {
          loss + maxMargin
        } else {
          loss
        }
    }
  }
}