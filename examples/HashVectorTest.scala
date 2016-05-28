import breeze.collection.mutable.OpenAddressHashArray
import breeze.linalg.{DenseVector, HashVector}
import org.apache.spark.mllib.linalg.Vectors

/**
 * Created by yuhao on 5/26/16.
 */
object HashVectorTest {

  def main(args: Array[String]) {

    val hashvec = new HashVector[Double](new OpenAddressHashArray[Double](5))

    val denseVec = Vectors.dense(Array(0.0, 0.1, 0.3, 0.0, 0.0))


    hashvec(2) = 2.5
    hashvec.array
    println(hashvec.activeIterator.mkString(", "))

    denseVec.foreachActive { case (index, value) => hashvec(index) = value + hashvec(index) }
    println(hashvec.activeIterator.mkString(", "))

    println(new OpenAddressHashArray[Double](5).isInstanceOf[Serializable])

  }

}
