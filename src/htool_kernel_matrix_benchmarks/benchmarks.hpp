#ifndef HTOOL_KERNEL_MATRIX_BENCHMARKS_CPP
#define HTOOL_KERNEL_MATRIX_BENCHMARKS_CPP

#include "kernel.hpp"
#include <htool/htool.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

namespace py = pybind11;
using namespace htool;
template <typename... Args>
using overload_cast_ = pybind11::detail::overload_cast_impl<Args...>;

template <typename T, typename ClusteringType>
class Benchmark {
  private:
    int spatial_dimension;
    std::string KernelType;
    std::string CompressorType;

    std::shared_ptr<Cluster<ClusteringType>> source_cluster, target_cluster;
    std::unique_ptr<VirtualKernel<T>> generator;
    std::shared_ptr<VirtualLowRankGenerator<T>> compressor;
    std::unique_ptr<HMatrix<T>> HA;

  public:
    Benchmark(int spatial_dimension0, std::string KernelType0, std::string CompressorType0) : spatial_dimension(spatial_dimension0), source_cluster(new Cluster<ClusteringType>(spatial_dimension)), KernelType(KernelType0), CompressorType(CompressorType0), target_cluster(new Cluster<ClusteringType>(spatial_dimension)){};
    void build_clusters(int nb_rows, int nb_cols, const py::array_t<double, py::array::f_style | py::array::forcecast> &target_points, const py::array_t<double, py::array::f_style | py::array::forcecast> &source_points) {
        source_cluster->build(nb_cols, source_points.data(), 2);
        target_cluster->build(nb_rows, target_points.data(), 2);
        if (this->KernelType == "InverseDistanceKernel")
            generator = std::unique_ptr<InverseDistanceKernel<T>>(new InverseDistanceKernel<T>(spatial_dimension, nb_rows, nb_cols, target_points.data(), source_points.data()));
        else if (this->KernelType == "GaussianKernel")
            generator = std::unique_ptr<GaussianKernel<T>>(new GaussianKernel<T>(spatial_dimension, nb_rows, nb_cols, target_points.data(), source_points.data()));
        else {
            throw std::logic_error("Kernel type not supported");
        }

        if (this->CompressorType == "partialACA")
            compressor = std::shared_ptr<partialACA<T>>();
        else {
            throw std::logic_error("Compressor type not supported");
        }
    };
    void build_clusters(int nb_rows, const py::array_t<double, py::array::f_style | py::array::forcecast> &target_points) {
        target_cluster->build(nb_rows, target_points.data(), 2);
        source_cluster = target_cluster;

        if (this->KernelType == "InverseDistanceKernel")
            generator = std::unique_ptr<InverseDistanceKernel<T>>(new InverseDistanceKernel<T>(spatial_dimension, nb_rows, target_points.data()));
        else if (this->KernelType == "GaussianKernel")
            generator = std::unique_ptr<GaussianKernel<T>>(new GaussianKernel<T>(spatial_dimension, nb_rows, target_points.data()));
        else {
            throw std::logic_error("Kernel type not supported");
        }

        if (this->CompressorType == "partialACA")
            compressor = std::shared_ptr<partialACA<T>>();
        else {
            throw std::logic_error("Compressor type not supported");
        }
    };
    void build_HMatrix(const py::array_t<double, py::array::f_style | py::array::forcecast> &target_points, const py::array_t<double, py::array::f_style | py::array::forcecast> &source_points, double epsilon, double eta, double mintargetdepth, double minsourcedepth, double maxblocksize) {
        HA = std::unique_ptr<HMatrix<T>>(new HMatrix<T>(target_cluster, source_cluster, epsilon, eta));
        HA->set_compression(compressor);
        HA->set_minsourcedepth(minsourcedepth);
        HA->set_mintargetdepth(mintargetdepth);
        HA->set_maxblocksize(maxblocksize);

        HA->build(*generator, target_points.data(), source_points.data());
    };

    void build_HMatrix(const py::array_t<double, py::array::f_style | py::array::forcecast> &target_points, double epsilon, double eta, double mintargetdepth, double minsourcedepth, double maxblocksize) {
        HA = std::unique_ptr<HMatrix<T>>(new HMatrix<T>(target_cluster, source_cluster, epsilon, eta));
        HA->set_compression(compressor);
        HA->set_minsourcedepth(minsourcedepth);
        HA->set_mintargetdepth(mintargetdepth);
        HA->set_maxblocksize(maxblocksize);

        HA->build(*generator, target_points.data());
    };

    void product(const py::array_t<T, py::array::f_style | py::array::forcecast> &source_signal, py::array_t<T, py::array::f_style | py::array::forcecast> &result) {
        int mu;
        if (source_signal.ndim() == 1) {
            mu = 1;
        } else if (source_signal.ndim() == 2) {
            mu = source_signal.shape()[1];
        }
        HA->mvprod_global_to_global(source_signal.data(), result.mutable_data(), mu);
    }

    void print_HMatrix_infos() {
        HA->print_infos();
    }
};

template <typename T, typename ClusteringType>
void declare_HtoolBenchmark(py::module &m, const std::string &className) {
    using Class = Benchmark<T, ClusteringType>;
    py::class_<Class> py_class(m, className.c_str());
    py_class.def(py::init<int, std::string, std::string>());
    py_class.def("build_clusters", overload_cast_<int, int, const py::array_t<double, py::array::f_style | py::array::forcecast> &, const py::array_t<double, py::array::f_style | py::array::forcecast> &>()(&Class::build_clusters));
    py_class.def("build_clusters", overload_cast_<int, const py::array_t<double, py::array::f_style | py::array::forcecast> &>()(&Class::build_clusters));
    py_class.def("build_HMatrix", overload_cast_<const py::array_t<double, py::array::f_style | py::array::forcecast> &, const py::array_t<double, py::array::f_style | py::array::forcecast> &, double, double, double, double, double>()(&Class::build_HMatrix));
    py_class.def("build_HMatrix", overload_cast_<const py::array_t<double, py::array::f_style | py::array::forcecast> &, double, double, double, double, double>()(&Class::build_HMatrix));
    py_class.def("product", &Class::product);
    py_class.def("print_HMatrix_infos", &Class::print_HMatrix_infos);
}

#endif
