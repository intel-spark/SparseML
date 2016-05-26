package org.apache.spark.mllib.classification

import org.apache.spark.SparkContext
import org.apache.spark.annotation.Since
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.mllib.classification.impl.GLMClassificationModel
import org.apache.spark.mllib.linalg.{DenseVector, Vector, Vectors}
import org.apache.spark.mllib.linalg.BLAS.dot
import org.apache.spark.mllib.optimization._
import org.apache.spark.mllib.pmml.PMMLExportable
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.util.{DataValidators, Loader, Saveable}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.storage.StorageLevel

class SparseLogisticRegressionWithLBFGS
  extends SparseGeneralizedLinearAlgorithm[LogisticRegressionModel] with Serializable {

  this.setFeatureScaling(true)

  @Since("1.1.0")
  override val optimizer = new SparseLBFGS(null, new SquaredL2Updater)

  override protected val validators = List(multiLabelValidator)

  private def multiLabelValidator: RDD[LabeledPoint] => Boolean = { data =>
    if (numOfLinearPredictor > 1) {
      DataValidators.multiLabelValidator(numOfLinearPredictor + 1)(data)
    } else {
      DataValidators.binaryLabelValidator(data)
    }
  }

  /**
   * Set the number of possible outcomes for k classes classification problem in
   * Multinomial Logistic Regression.
   * By default, it is binary logistic regression so k will be set to 2.
   */
  @Since("1.3.0")
  def setNumClasses(numClasses: Int): this.type = {
    require(numClasses > 1)
    numOfLinearPredictor = numClasses - 1
//    if (numClasses > 2) {
//      optimizer.setGradient(new SparseLogisticGradient(numClasses))
//    }
    this
  }

  override protected def createModel(weights: Vector, intercept: Double) = {
    if (numOfLinearPredictor == 1) {
      new LogisticRegressionModel(weights, intercept)
    } else {
      new LogisticRegressionModel(weights, intercept, numFeatures, numOfLinearPredictor + 1)
    }
  }

  /**
   * Run Logistic Regression with the configured parameters on an input RDD
   * of LabeledPoint entries.
   *
   * If a known updater is used calls the ml implementation, to avoid
   * applying a regularization penalty to the intercept, otherwise
   * defaults to the mllib implementation. If more than two classes
   * or feature scaling is disabled, always uses mllib implementation.
   * If using ml implementation, uses ml code to generate initial weights.
   */
  override def run(input: RDD[LabeledPoint]): LogisticRegressionModel = {
    run(input, generateInitialWeights(input), userSuppliedWeights = false)
  }

  /**
   * Run Logistic Regression with the configured parameters on an input RDD
   * of LabeledPoint entries starting from the initial weights provided.
   *
   * If a known updater is used calls the ml implementation, to avoid
   * applying a regularization penalty to the intercept, otherwise
   * defaults to the mllib implementation. If more than two classes
   * or feature scaling is disabled, always uses mllib implementation.
   * Uses user provided weights.
   *
   * In the ml LogisticRegression implementation, the number of corrections
   * used in the LBFGS update can not be configured. So `optimizer.setNumCorrections()`
   * will have no effect if we fall into that route.
   */
  override def run(input: RDD[LabeledPoint], initialWeights: Vector): LogisticRegressionModel = {
    run(input, initialWeights, userSuppliedWeights = true)
  }

  private def run(input: RDD[LabeledPoint], initialWeights: Vector, userSuppliedWeights: Boolean):
  LogisticRegressionModel = {
    // ml's Logistic regression only supports binary classification currently.
    if (numOfLinearPredictor == 1) {
//      def runWithMlLogisitcRegression(elasticNetParam: Double) = {
//        // Prepare the ml LogisticRegression based on our settings
//        val lr = new org.apache.spark.ml.classification.LogisticRegression()
//        lr.setRegParam(optimizer.getRegParam())
//        lr.setElasticNetParam(elasticNetParam)
//        lr.setStandardization(useFeatureScaling)
//        if (userSuppliedWeights) {
//          val uid = Identifiable.randomUID("logreg-static")
//          lr.setInitialModel(new org.apache.spark.ml.classification.LogisticRegressionModel(
//            uid, initialWeights.asML, 1.0))
//        }
//        lr.setFitIntercept(addIntercept)
//        lr.setMaxIter(optimizer.getNumIterations())
//        lr.setTol(optimizer.getConvergenceTol())
//        // Convert our input into a DataFrame
//        val sqlContext = new SQLContext(input.context)
//        import sqlContext.implicits._
//        val df = input.map(_.asML).toDF()
//        // Determine if we should cache the DF
//        val handlePersistence = input.getStorageLevel == StorageLevel.NONE
//        // Train our model
//        val mlLogisticRegresionModel = lr.train(df, handlePersistence)
//        // convert the model
//        val weights = Vectors.dense(mlLogisticRegresionModel.coefficients.toArray)
//        createModel(weights, mlLogisticRegresionModel.intercept)
//      }
//      optimizer.getUpdater() match {
//        case x: SquaredL2Updater => runWithMlLogisitcRegression(0.0)
//        case x: L1Updater => runWithMlLogisitcRegression(1.0)
//        case _ => super.run(input, initialWeights)
//      }
      super.run(input, initialWeights)
    } else {
      super.run(input, initialWeights)
    }
  }
}