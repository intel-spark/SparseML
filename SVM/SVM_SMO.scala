package ml.features

import org.apache.log4j.Level
import org.apache.log4j.Logger
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import _root_.scala.util.Random
import scala.util.Random

/**
 * Created by yuhao on 5/4/16.
 */
class SVM_SMO extends Serializable {

  import SVM_SMO._

  def train(dataset: RDD[LabeledPoint]): SVM_SMOModel = {

    val subModels = dataset.mapPartitions{ iter =>
      val model = train(iter.toArray)
      Iterator(model)
    }.collect()

    val count = subModels.length
    val w = subModels.map(_.weight).transpose.map(_.sum).map(d => d / count)
    val b = subModels.map(_.b).sum / count

    new SVM_SMOModel(w, b)
  }


  def train(dataset: Array[LabeledPoint]): SVM_SMOModel = {
    val data = dataset

    val labels = data.map(_.label)
    var C = 1.0; // C value. Decrease for more regularization
    var tol =  1e-4; // numerical tolerance. Don't touch unless you're pro
    var maxiter = 2; // max number of iterations
    var numpasses = 10; // how many passes over data with no change before we halt? Increase for more precision.

    // instantiate kernel according to options. kernel can be given as string or as a custom function
    val kernel = linearKernel
    var kernelType = "linear"

    val N = data.length
    val D = data(0).features.size
    val alpha = Array.fill[Double](N)(0)
    var b = 0.0

    var iter = 0
    var passes = 0
    def kernelResult = (i: Int, j: Int) => kernel(data(i).features, data(j).features)

    while(passes < numpasses && iter < maxiter) {
      var alphaChanged = 0
      for (i <- 0 until N) {
        val Errori = marginOne(data(i).features, alpha, data, b) - data(i).label
        if ((labels(i) * Errori < -tol && alpha(i) < C)
          || (labels(i) * Errori > tol && alpha(i) > 0)) {

          // alpha_i needs updating! Pick a j to update it with
          var j = i
          while (j == i) j = new Random().nextInt(N)
          val Ej = marginOne(data(j).features, alpha, data, b) - labels(j)

          // calculate L and H bounds for j to ensure we're in [0 C]x[0 C] box
          val ai = alpha(i)
          val aj = alpha(j)
          var L = 0D
          var H = C
          if (labels(i) == labels(j)) {
            L = math.max(0, ai + aj - C)
            H = Math.min(C, ai + aj)
          } else {
            L = Math.max(0, aj - ai)
            H = Math.min(C, C + aj - ai)
          }

          if (Math.abs(L - H) < 1e-4) {}
          else {
            var eta = 2 * kernelResult(i, j)
            eta = eta - kernelResult(i, i)
            eta = eta - kernelResult(j, j)
            if (eta < 0) {
              // compute new alpha_j and clip it inside [0 C]x[0 C] box
              // then compute alpha_i based on it.
              var newaj = aj - labels(j) * (Errori - Ej) / eta
              if (newaj > H) newaj = H
              if (newaj < L) newaj = L
              println(aj-newaj)
              if (Math.abs(aj - newaj) < 1e-9) {} else {
                alpha(j) = newaj
                val newai = ai + labels(i) * labels(j) * (aj - newaj)
                alpha(i) = newai

                // update the bias term
                val b1 = b - Errori - labels(i) * (newai - ai) * kernelResult(i, i)
                -labels(j) * (newaj - aj) * kernelResult(i, j)
                val b2 = b - Ej - labels(i) * (newai - ai) * kernelResult(i, j)
                -labels(j) * (newaj - aj) * kernelResult(j, j)
                b = 0.5 * (b1 + b2)
                if (newai > 0 && newai < C) b = b1
                if (newaj > 0 && newaj < C) b = b2
                alphaChanged += 1
              }
            }
          }
        }
      }

      iter+= 1
      //console.log("iter number %d, alphaChanged = %d", iter, alphaChanged);
      if(alphaChanged == 0) passes+= 1
      else passes= 0
    }
    val w = new Array[Double](D)
    for(j <- 0 until D) {
      var s= 0.0
      for(i <- 0 until N) {
        s+= alpha(i) * labels(i) * data(i).features(j)
      }
      w(j) = s
    }

    new SVM_SMOModel(w, b)
  }
}

object SVM_SMO  extends Serializable {

  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)
    val conf = new SparkConf().setAppName("ssss").setMaster("local[8]")
    val sc = new SparkContext(conf)
    val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")
      .map(p => new LabeledPoint( if(p.label == 0) -1.0 else 1.0, p.features)).repartition(8)

    val model = new SVM_SMO().train(data)

    println("total: " + data.collect().length)
    val correct = data.collect().map(d => if(model.predict(d.features) == d.label) 1 else 0).sum
    println(correct)
  }


  def linearKernel = (v1: Vector, v2: Vector) => {
    var s = 0D
    for(q <- 0 until v1.size) { s += v1(q) * v2(q) }
    s
  }

  def RBFKernal = (v1: Vector, v2: Vector) => {
    val sigma = 0.5
    var s=0
    for(q <- 0 until v1.size) { s += (v1(q) - v2(q)) * (v1(q) - v2(q)) }
    Math.exp(- s / (2.0 * sigma * sigma))
  }

  def marginOne(inst: Vector, alpha: Array[Double], data: Array[LabeledPoint], b: Double): Double =  {

    data.zip(alpha).map{ case (point, alp) =>
      point.label * alp * linearKernel(inst, point.features)
    }.sum + b
  }
}

class SVM_SMOModel (val weight: Array[Double], val b: Double)  extends Serializable {
  def predict(v: Vector): Double = {
    val value = v.toArray.zip(weight).map{ case (d1, d2) => d1 * d2 }.sum + b
    if(value > 0) 1 else -1
  }
}
