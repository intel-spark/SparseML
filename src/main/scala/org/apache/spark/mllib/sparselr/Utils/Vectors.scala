package org.apache.spark.mllib.sparselr.Utils

import it.unimi.dsi.fastutil.ints.Int2DoubleOpenHashMap
/**
 * Represents a numeric vector, whose index type is Int and value type is Double.
 *
 * Note: Users should not implement this interface.
 */
sealed trait Vector extends Serializable {

  def iterator: Iterator[(Int, Double)]

  /**
   * Converts the instance to a double array.
   */
  def toArray: Array[Double] = {
    throw new NotImplementedError(s"copy is not implemented for ${this.getClass}.")
  }

  def copy: Vector = {
    throw new NotImplementedError(s"copy is not implemented for ${this.getClass}.")
  }

  def apply(i: Int): Double = {
    throw new NotImplementedError(s"apply is not implemented for ${this.getClass}.")
  }

  def update(i: Int, value: Double): Unit = {
    throw new NotImplementedError(s"update is not implemented for ${this.getClass}.")
  }
}

/**
 * Factory methods for [[org.apache.spark.mllib.sparselr.Utils.Vector]].
 */
object Vectors {

  /**
   * Creates a dense vector from a double array.
   */
  def dense(values: Array[Double]): Vector = new DenseVector(values)

  /**
   * Creates a hashedsparse vector.
   */
  def hashSparseVector(): Vector = new HashedSparseVector
}

/**
 * A dense vector represented by a value array.
 */
class DenseVector(val values: Array[Double]) extends Vector {
  override def toArray: Array[Double] = values

  override def iterator: Iterator[(Int, Double)] =  new Iterator[(Int, Double)] {
    private var pos = 0
    private var index = 0
    private var value = 0.0
    
    override def hasNext: Boolean = pos < values.length
    
    override def next(): (Int, Double) = {
      index = pos
      value = values(pos)
      pos += 1
      (index, value)
    }
  }
  
  override def copy: DenseVector = {
    new DenseVector(values.clone())
  }

  override def apply(i: Int): Double = values(i)

  override def update(i: Int, value: Double): Unit = {
   values(i) = value
  }
}

/**
 * A sparse vector represented by an index array and an value array.
 */
class HashedSparseVector() extends Vector {
  private val _hashmap: Int2DoubleOpenHashMap = new Int2DoubleOpenHashMap()

  def hashmap: Int2DoubleOpenHashMap = _hashmap

  def contains(i: Int): Boolean = _hashmap.containsKey(i)

  override def apply(i: Int): Double = _hashmap.get(i)

  override def update(i: Int, value: Double): Unit = {
    _hashmap.put(i, value)
  }

//  override def toArray: Array[Double] = {
//    val iter = _hashmap.int2DoubleEntrySet.fastIterator()
//
//    val data = new PrimitiveVector[Double]
//    while (iter.hasNext())
//    { val entry = iter.next()
//      data(entry.getIntKey) = entry.getDoubleValue
//    }
//    data.trim.array
//  }

  override def iterator: Iterator[(Int, Double)] = new Iterator[(Int, Double)] {
    val iter = _hashmap.int2DoubleEntrySet.fastIterator()

    override def hasNext: Boolean = iter.hasNext

    override def next(): (Int, Double) = {
      val entry = iter.next()
      (entry.getIntKey, entry.getDoubleValue)
    }
  }
}

class SparseVector(
          var indices: Array[Int],
          var values: Array[Float]
              ) extends Vector {

  override def iterator: Iterator[(Int, Double)] = new Iterator[(Int, Double)] {
    private var pos = 0

    private var index = 0
    private var value = 0.0

    override def hasNext: Boolean = pos < indices.length

    override def next(): (Int, Double) = {
      index = indices(pos)
      value = values(pos)
      pos += 1
      (index, value)
    }
  }

  override def copy: SparseVector = {
    new SparseVector(indices.clone(), values.clone())
  }
}

private class CompressedSparseVector(
  var indices: Array[Byte],
  var values: Array[Float],
  var binaryIndices: Array[Byte],
  var indicesBeginPos: Int,
  var indicesEndPos: Int,
  var floatValuesBeginPos: Int,
  var floatValuesEndPos: Int,
  var binaryIndicesBeginPos: Int,
  var binaryIndicesEndPos: Int,
//  var mappingsPerPartition: PrimitiveVector[Int]=null ) extends Vector{
  var mappingsPerPartition: Array[Int]=null ) extends Vector{

//  override def toArray: Array[Double] = {
//    val data = new PrimitiveVector[Double]()
//    var pos = binaryIndicesBeginPos
//    var valueIndex = floatValuesBeginPos
//    var globalKey = 0
//
//    while(pos < binaryIndicesEndPos) {
//      val (key, nextPos) = LRUtils.bytes2Int(binaryIndices, pos)
//      globalKey = mappingsPerPartition(key)
//      pos = nextPos + 1
//      data(globalKey) = 1.0
//    }
//
//    pos = indicesBeginPos
//    while(pos < indicesEndPos && valueIndex < floatValuesEndPos) {
//      val (key, nextPos) = LRUtils.bytes2Int(indices, pos)
//      globalKey = mappingsPerPartition(key)
//      data(globalKey) = values(valueIndex)
//      pos = nextPos + 1
//      valueIndex += 1
//    }
//    assert(pos==indicesEndPos && valueIndex==floatValuesEndPos)
//    data.trim.array
//  }

  override def iterator: Iterator[(Int, Double)] =  new Iterator[(Int, Double)] {
    private var floatValuePos = floatValuesBeginPos
    private var visitedBinaryIndices: Boolean = if(binaryIndicesEndPos == binaryIndicesBeginPos) true else false
    private var pos = if(!visitedBinaryIndices) binaryIndicesBeginPos else indicesBeginPos
    
    private var index = 0
    private var value = 0.0

    override def hasNext: Boolean = if (!visitedBinaryIndices) (pos < binaryIndicesEndPos) else (pos < indicesEndPos)

    override def next(): (Int, Double) = {
      if (!visitedBinaryIndices) {
        val (key, endPos) = LRUtils.bytes2Int(binaryIndices, pos)
        value = 1.0
        index = mappingsPerPartition(key)
        pos = endPos + 1
        if (pos >= binaryIndicesEndPos) {
          pos = indicesBeginPos
          visitedBinaryIndices = true
        }
      } else {
        val (key, endPos) = LRUtils.bytes2Int(indices, pos)
        index = mappingsPerPartition(key)
        value = values(floatValuePos)
        floatValuePos += 1
        pos = endPos + 1
      }

      (index, value)
    }
  }
  
  override def copy: CompressedSparseVector = {
    new CompressedSparseVector(indices.clone(), values.clone(), binaryIndices.clone(),
      indicesBeginPos, indicesEndPos,floatValuesBeginPos, floatValuesEndPos,
      binaryIndicesBeginPos, binaryIndicesEndPos, mappingsPerPartition.clone)
  }
}