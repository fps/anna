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

  Eigen::Matrix<float, 16, cols> m3;
  Eigen::Matrix<float, 16, 64> n3;
  Eigen::Matrix<float, 16, 64> o3;

  Eigen::Matrix<float, 16, cols> m4;
  Eigen::Matrix<float, 16, 64> n4;
  Eigen::Matrix<float, 16, 64> o4;

  Eigen::Matrix<float, 16, cols> m5;
  Eigen::Matrix<float, 16, 64> n5;
  Eigen::Matrix<float, 16, 64> o5;

  for (auto _ : state)
  {
    n = m.middleCols(cols - 64, 64) * 2.0f + m.middleCols(64, 64) * 2.0f + o;
    n2 = m2.middleCols(cols - 64, 64) * 2.0f + m.middleCols(64, 64) * 2.0f + o;
    n3 = m3.middleCols(cols - 64, 64) * 2.0f + m.middleCols(64, 64) * 2.0f + o;
    n4 = m4.middleCols(cols - 64, 64) * 2.0f + m.middleCols(256, 64) * 2.0f + o;
    n5 = m5.middleCols(cols - 64, 64) * 2.0f + m.middleCols(0, 64) * 2.0f + o;
    benchmark::DoNotOptimize(&n);
  }
}

BENCHMARK(run<1 << 9>); 
BENCHMARK(run<1 << 10>); 
BENCHMARK(run<1 << 11>); 
BENCHMARK(run<1 << 13>); 
BENCHMARK(run<1 << 14>); 
BENCHMARK_MAIN();

