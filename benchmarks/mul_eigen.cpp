#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include <anna/benchmark.hpp>

template<int rows, int cols, int n>
void eigen(benchmark::State & state)
{
  Eigen::Matrix<float, rows, n> r = Eigen::Matrix<float, rows, n>::Zero();
  Eigen::Matrix<float, rows, cols> m = Eigen::Matrix<float, rows, cols>::Ones();
  Eigen::Matrix<float, rows, n> v = Eigen::Matrix<float, cols, n>::Ones();
  
  for (auto _ : state)
  {
    r = m * v;

    escape(&r);
  }
}


BENCHMARK(eigen<64, 64, 1024>);
BENCHMARK(eigen<32, 32, 1024>);
BENCHMARK(eigen<16, 16, 1024>);
BENCHMARK(eigen<8, 8, 1024>);

BENCHMARK(eigen<64, 64, 128>);
BENCHMARK(eigen<32, 32, 128>);
BENCHMARK(eigen<16, 16, 128>);
BENCHMARK(eigen<8, 8, 128>);

BENCHMARK(eigen<64, 64, 64>);
BENCHMARK(eigen<32, 32, 64>);
BENCHMARK(eigen<16, 16, 64>);
BENCHMARK(eigen<8, 8, 64>);

BENCHMARK(eigen<64, 64, 1>);
BENCHMARK(eigen<32, 32, 1>);
BENCHMARK(eigen<16, 16, 1>);
BENCHMARK(eigen<8, 8, 1>);

BENCHMARK_MAIN();
