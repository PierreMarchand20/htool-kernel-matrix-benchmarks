#include "benchmarks.hpp"
#include "kernel.hpp"
#include "wrapper_mpi.hpp"

PYBIND11_MODULE(HtoolKernelMatrixBenchmarks, m) {
    // import the mpi4py API
    if (import_mpi4py() < 0) {
        throw std::runtime_error("Could not load mpi4py API."); // LCOV_EXCL_LINE
    }

    declare_HtoolBenchmark<double, htool::PCARegularClustering>(m, "HtoolBenchmarkPCARegularClusteringDouble");
    declare_HtoolBenchmark<float, htool::PCARegularClustering>(m, "HtoolBenchmarkPCARegularClusteringFloat");
    // declare_HtoolBenchmark<std::complex<double>, htool::PCARegularClustering>(m, "HtoolBenchmarkPCARegularClusteringComplex");
}
