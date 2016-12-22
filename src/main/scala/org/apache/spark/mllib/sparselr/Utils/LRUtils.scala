package org.apache.spark.mllib.sparselr.Utils

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

object LRUtils {
  /**
   * When `x` is positive and large, computing `math.log(1 + math.exp(x))` will lead to arithmetic
   * overflow. This will happen when `x > 709.78` which is not a very large number.
   * It can be addressed by rewriting the formula into `x + math.log1p(math.exp(-x))` when `x > 0`.
   *
   * @param x a floating-point value as input.
   * @return the result of `math.log(1 + math.exp(x))`.
   */
  def log1pExp(x: Double): Double = {
    if (x > 0) {
      x + math.log1p(math.exp(-x))
    } else {
      math.log1p(math.exp(x))
    }
  }

  /**
   * Writes the specified int to the buffer using 1 to 5 bytes, depending on the size of the number.
   * @param number Int which needs to be serialized into bytes
   * @return the number of bytes written.
   */
  def int2Bytes(number: Int): Array[Byte] = {
    var value2 = number
    var value3 = value2 >> 7

    val buffer = new PrimitiveVector[Byte](5)

    if(value2 == 0) {
      buffer += value2.toByte
    }

    while(value2 > 0) {
      if (value3 > 0) {
        buffer += ((value2 & 0x7F) | 0x80).toByte
      } else {
        buffer += (value2 & 0x7F).toByte
      }
      value2 = value3
      value3 >>= 7
    }
    return buffer.trim.array
  }

  /**
   * Reads an int from the buffer.
   * @param buffer array which holds serialized bytes
   * @param pos start position of buffer
   * @return (number, position). number is integer deserialized from buffer. position is the end position of the number
   */
  def bytes2Int (buffer: Array[Byte], pos: Int): (Int, Int) = {
    var result: Int = 0
    var position: Int = pos
    var byte = buffer(pos)
    var shiftNum = 0

    while ((byte & 0x80) != 0) {
      result = result | ((byte & 0x7F)<<shiftNum)
      position += 1
      byte = buffer(position)
      shiftNum += 7
    }
    result = result | ((byte & 0x7F)<<shiftNum)
    (result, position)
  }

  //featureId cached in X is localId
  def loadFileAsMatrix(
                sc: SparkContext,
                path: String,
                minPartitions: Int): RDD[(Array[Double], Matrix)] = {
    val lines = sc.textFile(path, minPartitions)
      .map(_.trim)
      .filter(line => !(line.isEmpty || line.startsWith("#")))

    val data = lines.mapPartitions { samples =>
      val labels = new PrimitiveVector[Double]()
      val builder = new MatrixBuilder()

      samples.foreach { line =>
        val items = line.split(' ')

        labels += items.head.toDouble

        val featureIdAndValues = items.tail.filter(_.nonEmpty)

        val indices = new PrimitiveVector[Int]()
        val values = new PrimitiveVector[Float]()
        featureIdAndValues.foreach { item =>
          val featureAndValue = item.split(":")
          indices += featureAndValue(0).toInt
          val value = featureAndValue(1).toFloat
          values += value
        }
        builder.add(new SparseVector(indices.trim.array, values.trim.array))
      }
      Iterator((labels.trim.array, builder.toMatrix))
    }
    data
  }
}