package org.apache.spark.mllib.sparselr.Utils

import java.util.NoSuchElementException
import scala.reflect.ClassTag

/**
 * An append-only, non-threadsafe, array-backed vector that is optimized for primitive types.
 */
private[sparselr]
class PrimitiveVector[@specialized(Byte, Int, Double) V: ClassTag](initialSize: Int = 64)
  extends Serializable{
  private var _numElements = 0
  private var _array: Array[V] = _

  // NB: This must be separate from the declaration, otherwise the specialized parent class
  // will get its own array with the same initial size.
  _array = new Array[V](initialSize)

  def apply(index: Int): V = {
    require(index < _numElements)
    _array(index)
  }

  def update(index: Int, value: V): Unit = {
    _array(index) = value
  }

  def +=(value: V): Unit = {
    if (_numElements == _array.length) {
      resize(_array.length * 2)
    }
    _array(_numElements) = value
    _numElements += 1
  }

  def capacity: Int = _array.length

  def length: Int = _numElements

  def size: Int = _numElements

  def iterator: Iterator[V] = new Iterator[V] {
    var index = 0
    override def hasNext: Boolean = index < _numElements
    override def next(): V = {
      if (!hasNext) {
        throw new NoSuchElementException
      }
      val value = _array(index)
      index += 1
      value
    }
  }

  /** Gets the underlying array backing this vector. */
  def array: Array[V] = _array

  /** Trims this vector so that the capacity is equal to the size. */
  def trim(): PrimitiveVector[V] = resize(size)

  /** Resizes the array, dropping elements if the total length decreases. */
  def resize(newLength: Int): PrimitiveVector[V] = {
    val newArray = new Array[V](newLength)
    _array.copyToArray(newArray)
    _array = newArray
    if (newLength < _numElements) {
      _numElements = newLength
    }
    this
  }
}

