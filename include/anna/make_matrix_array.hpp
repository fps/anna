#pragma once

#include <Eigen/Core>
#include <array>

namespace anna
{
  template<typename T, int Rows, int Cols, int K>
  std::array<Eigen::Matrix<T, Rows, Cols>, K>
  make_matrix_array(T value)
  {
    std::array<Eigen::Matrix<T, Rows, Cols>, K> w;
    w.fill(Eigen::Matrix<T, Rows, Cols>::Constant(value));
    return w;
  }
} // namespace nn
