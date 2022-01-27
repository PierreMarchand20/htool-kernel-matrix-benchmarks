#ifndef HTOOL_KERNEL_MATRIX_BENCHMARKS_KERNEL_CPP
#define HTOOL_KERNEL_MATRIX_BENCHMARKS_KERNEL_CPP

#include <htool/types/virtual_generator.hpp>

template <typename T>
class VirtualKernel : public htool::VirtualGenerator<T> {
  protected:
    const double *target_points;
    const double *source_points;
    int spatial_dimension;

  public:
    VirtualKernel(int spatial_dimension0, int nr, int nc, const double *p10, const double *p20) : htool::VirtualGenerator<T>(nr, nc), target_points(p10), source_points(p20), spatial_dimension(spatial_dimension0) {}

    VirtualKernel(int spatial_dimension0, int nr, const double *p10) : htool::VirtualGenerator<T>(nr, nr), target_points(p10), source_points(p10), spatial_dimension(spatial_dimension0) {}
};

template <typename T>
class LaplacianKernel : public VirtualKernel<T> {
  public:
    using VirtualKernel<T>::VirtualKernel;
    void copy_submatrix(int M, int N, const int *const rows, const int *const cols, T *ptr) const override {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                ptr[i + M * j] = 1. / (1e-5 + 4 * M_PI * std::sqrt(std::inner_product(this->target_points + this->spatial_dimension * i, this->target_points + this->spatial_dimension * i + this->spatial_dimension, this->source_points + this->spatial_dimension * j, double(0), std::plus<double>(), [](double u, double v) { return (u - v) * (u - v); })));
            }
        }
    }
};

#endif
