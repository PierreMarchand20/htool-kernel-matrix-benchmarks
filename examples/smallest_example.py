
import HtoolKernelMatrixBenchmarks as HtoolBench
import numpy as np
import mpi4py


# Random geometry
NbRows = 500
NbCols = 500
np.random.seed(0)
points_target = np.zeros((2, NbRows))
points_target[0, :] = np.random.random(NbRows)
points_target[1, :] = np.random.random(NbRows)

if NbRows == NbCols:
    points_source = points_target
else:
    points_source = np.zeros((2, NbCols))
    points_source[0, :] = np.random.random(NbCols)
    points_source[1, :] = np.random.random(NbCols)

# Htool parameters
eta = 10
epsilon = 1e-1
minclustersize = 10
mintargetdepth = 0
minsourcedepth = 0
maxblocksize = 1000000

bench = HtoolBench.HtoolBenchmarkPCARegularClustering(
    3, "Laplacian", "partialACA")


bench.build_clusters(NbRows, NbCols, points_target, points_source)
bench.build_HMatrix(points_target, points_source, epsilon, eta,
                    mintargetdepth, minsourcedepth, maxblocksize)

x = np.random.rand(NbCols)
y = np.zeros(NbRows)
bench.product(x, y)

bench.print_HMatrix_infos()
