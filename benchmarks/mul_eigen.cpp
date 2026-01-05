#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include <anna/benchmark.hpp>

template<int rows, int cols, int n>
void mul(benchmark::State & state)
{
  Eigen::Matrix<float, rows, n> r = Eigen::Matrix<float, rows, n>::Zero();
  Eigen::Matrix<float, rows, cols> m = Eigen::Matrix<float, rows, cols>::Ones();
  Eigen::Matrix<float, cols, n> v = Eigen::Matrix<float, cols, n>::Ones();
  
  for (auto _ : state)
  {
    r = m * v;

    escape(&r);
  }
}

#include "mul.cpp"
