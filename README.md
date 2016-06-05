# SparseML

Yuhao Yang (yuhao.yang@intel.com)
<br><br>
From purchase history to movie ratings, data sparsity has always been one of the primary
characteristics of big data. Powerful as Spark is on parallel processing for the partitioned
data, many of the algorithms in MLlib are implemented based on the assumption of certain degree
of data density, such like the gradients of logistic regression, or cluster centers of KMeans.
Yet during collaboration with some Spark users, we often find their feature number at the
dimension of millions or even billions, which far exceeds the capacity of some important algorithms
in MLlib, or become impractical due to enormous memory consumption even with great sparsity in the
training data. To fill the gap, we present a Spark package containing some major improvements we
have conducted to support the sparse data at large scope. Through optimization on data structure,
network communication and arithmetic operation, we can extensively compress the memory consumption
and reduce computation cost for sparse data, thus to enable the algorithms on larger feature
dimensions and scope. Two of the examples are the successful support of our implementation on
logistic regression with 1 billion features and KMeans with 10M features and hundreds of clusters.
We’ll also share some work we are contributing to Spark and some best practices we have accumulated
in the context of sparse data support on Spark MLlib.


## Usage:
The class/function signature remains the same as in Spark MLlib. Please refer to the examples folder

## Performance:
Although the concrete performance improvements depends on the sparsity of the dataset. The algorithms
in SparseSpark generally significantly reduce the time and memory consumption compared with the original
Spark implementation.


## Accuracy
The optimization does not affect the accuracy. It yields the same result with the Spark version,
yet with less computation resources.