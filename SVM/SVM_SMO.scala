package ml.features

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

/**
 * Created by yuhao on 5/4/16.
 */
class SVM_SMO extends Serializable {

  import ml.features.SVM_SMO._

  private var kernelType = "linear"

  def setKernelType(value: String): this.type = {
    kernelType = value
    this
  }

  def train(dataset: RDD[LabeledPoint]): SVMModel = {

    val subModels = dataset.mapPartitions{ iter =>
      val model = train(iter.toArray)
      Iterator(model)
    }.collect()

    if (subModels.head.isInstanceOf[SVM_LinearModel]) {
      val count = subModels.length
      val w = subModels.map(_.asInstanceOf[SVM_LinearModel].weight).transpose.map(_.sum).map(d => d / count)
      val b = subModels.map(_.asInstanceOf[SVM_LinearModel].b).sum / count
      new SVM_LinearModel(w, b)
    }
    else {
      val count = subModels.length
      val alphaArray = new ArrayBuffer[Double]
      val alpha = subModels.map(_.asInstanceOf[SVM_RBFModel].alpha).foreach( arr => alphaArray ++= arr)
      val dataArray = new ArrayBuffer[LabeledPoint]
      val vectors = subModels.map(_.asInstanceOf[SVM_RBFModel].data).foreach( arr => dataArray ++= arr)

      val b = subModels.map(_.asInstanceOf[SVM_RBFModel].b).sum / count
      new SVM_RBFModel(alphaArray.toArray, dataArray.toArray, b)
    }
  }


  def train(dataset: Array[LabeledPoint]): SVMModel = {
    val data = dataset

    val labels = data.map(_.label)
    val C = 1.0; // C value. Decrease for more regularization
    val tol =  1e-4; // numerical tolerance. Don't touch unless you're pro
    val maxIter = 2; // max number of iterations
    val numPasses = 10; // how many passes over data with no change before we halt? Increase for more precision.

    // instantiate kernel according to options. kernel can be given as string or as a custom function
    val kernel = if(kernelType == "linear") linearKernel else RBFKernal

    val N = data.length
    val D = data(0).features.size
    val alpha = Array.fill[Double](N)(0)
    var b = 0.0

    var iter = 0
    var passes = 0
    def kernelResult = (i: Int, j: Int) => kernel(data(i).features, data(j).features)

    while(passes < numPasses && iter < maxIter) {
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


    if(kernelType == "linear"){
      val w = new Array[Double](D)
      for(j <- 0 until D) {
        var s= 0.0
        for(i <- 0 until N) {
          s+= alpha(i) * labels(i) * data(i).features(j)
        }
        w(j) = s
      }
      new SVM_LinearModel(w, b)
    }
    else{
      // okay, we need to retain all the support vectors in the training data,
      // we can't just get away with computing the weights and throwing it out

      // But! We only need to store the support vectors for evaluation of testing
      // instances. So filter here based on this.alpha[i]. The training data
      // for which this.alpha[i] = 0 is irrelevant for future.
      val supportingVectors = data.zip(alpha).filter(p => p._2 > 1e-7)
      new SVM_RBFModel(supportingVectors.map(_._2), supportingVectors.map(_._1), b)
    }
  }
}

object SVM_SMO  extends Serializable {

  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)
    val conf = new SparkConf().setAppName("ssss").setMaster("local[8]")
    val sc = new SparkContext(conf)
    val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")
      .map(p => new LabeledPoint( if(p.label == 0) -1.0 else 1.0, p.features)).repartition(4)

    val model = new SVM_SMO()
      .setKernelType("rbf")
      .train(data)

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
    var s = 0D
    for(q <- 0 until v1.size) { s += (v1(q) - v2(q)) * (v1(q) - v2(q)) }
    Math.exp(- s / (2.0 * sigma * sigma))
  }

  def marginOne(inst: Vector, alpha: Array[Double], data: Array[LabeledPoint], b: Double): Double =  {

    data.zip(alpha).map{ case (point, alp) =>
      point.label * alp * linearKernel(inst, point.features)
    }.sum + b
  }
}

class SVM_LinearModel (val weight: Array[Double], val b: Double) extends SVMModel with Serializable {
  def predict(v: Vector): Double = {
    val value = v.toArray.zip(weight).map{ case (d1, d2) => d1 * d2 }.sum + b
    if(value > 0) 1 else -1
  }
}

class SVM_RBFModel (val alpha: Array[Double], val data: Array[LabeledPoint], val b: Double) extends SVMModel with Serializable {
  def predict(v: Vector): Double = {
    val value = SVM_SMO.marginOne(v, alpha, data, b)
    if(value > 0) 1 else -1
  }
}

abstract class SVMModel {
  def predict(v: Vector): Double
}