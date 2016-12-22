package org.apache.spark.mllib.sparselr.Utils

import it.unimi.dsi.fastutil.ints.Int2IntOpenHashMap

class MatrixBuilder(size: Int = 64) {
  val hdfs2localIndex = new Int2IntOpenHashMap()
  var local2hdfsIndex = new PrimitiveVector[Int]()
  val indices = new PrimitiveVector[Byte]()
  val values = new PrimitiveVector[Float]()
  val binaryIndices = new PrimitiveVector[Byte]()
  val indicesPos = new PrimitiveVector[Int]()
  val valuesPos = new PrimitiveVector[Int]()
  val binaryIndicesPos = new PrimitiveVector[Int]()

  indicesPos += 0
  valuesPos += 0
  binaryIndicesPos += 0

  def add(sample: Vector): Unit = {
    sample.iterator.foreach { indexAndValue =>
      val index = indexAndValue._1
      val value = indexAndValue._2.toFloat
      if (value != 0) {
        if (!hdfs2localIndex.containsKey(index)) {
          hdfs2localIndex.put(index, local2hdfsIndex.size)
          local2hdfsIndex += index
        }
        if (value == 1) {
          LRUtils.int2Bytes(hdfs2localIndex.get(index)).foreach { binaryFeatureByte => binaryIndices += binaryFeatureByte }
        } else {
          val localIndexes = LRUtils.int2Bytes(hdfs2localIndex.get(index))
          localIndexes.foreach { featureByte => indices += featureByte }
          values += value
        }
      }
    }
    indicesPos += indices.size
    valuesPos += values.size
    binaryIndicesPos += binaryIndices.size
  }

  def toMatrix: Matrix = {
    new CompressedSparseMatrix(indices.trim.array, values.trim.array, binaryIndices.trim.array,
      local2hdfsIndex.trim.array, indicesPos.trim.array, valuesPos.trim.array, binaryIndicesPos.trim.array)
  }
}
