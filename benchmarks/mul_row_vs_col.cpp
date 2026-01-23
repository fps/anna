#include <benchmark/benchmark.h>
#include <anna/benchmark.hpp>
#include <Eigen/Core>

template<int O1, int O2>
static void mul(benchmark::State & state)
{
  Eigen::Matrix<float, 16, 16, O1> m;
  Eigen::Matrix<float, 16, 1024, O2> n;

  for (auto _ : state)
  {
    auto r = m * n;
    escape(&r);
  }
}

BENCHMARK(mul<Eigen::RowMajor, Eigen::RowMajor>);
BENCHMARK(mul<Eigen::ColMajor, Eigen::ColMajor>);
BENCHMARK(mul<Eigen::RowMajor, Eigen::ColMajor>);
BENCHMARK(mul<Eigen::ColMajor, Eigen::RowMajor>);
BENCHMARK_MAIN();
