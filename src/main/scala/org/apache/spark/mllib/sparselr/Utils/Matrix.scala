package org.apache.spark.mllib.sparselr.Utils

sealed trait Matrix extends Serializable {
  def copy: Matrix = {
    throw new NotImplementedError(s"copy is not implemented for ${this.getClass}.")
  }

  def iterator: Iterator[Vector]
}

class CompressedSparseMatrix(
                              var indices: Array[Byte],
                              var values: Array[Float],
                              var binaryIndices: Array[Byte],
                              var mappings: Array[Int],
                              var indicesPos: Array[Int],
                              var valuesPos: Array[Int],
                              var binaryIndicesPos: Array[Int]) extends Matrix {

  override def copy: CompressedSparseMatrix = {
    new CompressedSparseMatrix(indices.clone(), values.clone(),
      binaryIndices.clone(), mappings.clone(), indicesPos.clone(),
      valuesPos.clone(), binaryIndicesPos.clone())
  }

  override def iterator: Iterator[Vector] = new Iterator[Vector] {
    private var pos = 0

    override def hasNext: Boolean = (pos < indicesPos.length-1 ||
                                    pos < binaryIndicesPos.length-1)

    override def next(): Vector = {
      val x: Vector = new CompressedSparseVector(indices, values,
                          binaryIndices, indicesPos(pos), indicesPos(pos+1),
                          valuesPos(pos), valuesPos(pos+1), binaryIndicesPos(pos),
                          binaryIndicesPos(pos+1), mappings.array)
      pos += 1
      x
    }
  }

  def tupletIterator(lables: Array[Double]): Iterator[(Double, Vector)] = new Iterator[(Double, Vector)] {
    private var pos = 0
    override def hasNext: Boolean = (pos < indicesPos.length-1 || pos < binaryIndicesPos.length-1)

    override def next(): (Double, Vector) = {
      val x: Vector = new CompressedSparseVector(indices, values,
                          binaryIndices, indicesPos(pos), indicesPos(pos+1),
                          valuesPos(pos), valuesPos(pos+1), binaryIndicesPos(pos),
                          binaryIndicesPos(pos+1), mappings.array)
      val label = lables(pos)
      pos += 1

      ((label, x))
    }
  }
}