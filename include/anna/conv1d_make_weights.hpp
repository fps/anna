#pragma once

#include <Eigen/Core>
#include <array>

namespace anna
{
  template<typename T, int k, int inp, int outp>
  std::array<Eigen::Matrix<T, outp, inp>, k>
  make_weights(T value)
  {
    std::array<Eigen::Matrix<T, outp, inp>, k> w;
    w.fill(Eigen::Matrix<T, outp, inp>::Constant(value));
    return w;
  }
} // namespace nn
