#include <benchmark/benchmark.h>
#include <Eigen/Core>

template<int cols>
static inline void run(benchmark::State & state)
{
  Eigen::Matrix<float, 16, cols> m;
  Eigen::Matrix<float, 16, 64> n;
  Eigen::Matrix<float, 16, 64> o;

  Eigen::Matrix<float, 16, cols> m2;
  Eigen::Matrix<float, 16, 64> n2;
  Eigen::Matrix<float, 16, 64> o2;

  for (auto _ : state)
  {
    n = m.middleCols(cols - 64, 64) * 2.0f + m.middleCols(cols - 128, 64) * 2.0f + o;
    n2 = m2.middleCols(cols - 64, 64) * 2.0f + o;
    benchmark::DoNotOptimize(&n);
  }
}

BENCHMARK(run<1 << 8>); 
BENCHMARK(run<1 << 9>); 
BENCHMARK(run<1 << 10>); 
BENCHMARK(run<1 << 11>); 
BENCHMARK(run<1 << 13>); 
BENCHMARK(run<1 << 14>); 
BENCHMARK_MAIN();

