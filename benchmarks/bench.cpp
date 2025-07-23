#include <benchmark/benchmark.h>
#include <Eigen/Core>

static void escape(void *p)
{
  asm volatile("" : : "g"(p) : "memory");
}

static void eigen_mul16(benchmark::State & state)
{
  Eigen::Matrix<float, 16, 1> r = Eigen::Matrix<float, 16, 1>::Zero();
  Eigen::Matrix<float, 16, 16> m = Eigen::Matrix<float, 16, 16>::Ones();
  Eigen::Matrix<float, 16, 1> v = Eigen::Matrix<float, 16, 1>::Ones();
  
  for (auto _ : state)
  {
    r = m * v;

    escape(&r);
  }
}

static void raw_mul16(benchmark::State & state)
{
  float r[16];
  float m[16*16];
  float v[16];

  for (auto _ : state)
  {
    for (size_t row = 0; row < 16; ++row)
    {
      r[row] = 0;
      for (size_t col = 0; col < 16; ++col)
      {
        r[row] += m[col*16+row] * v[row];
      }
    }
    escape(r);
  }
}


BENCHMARK(eigen_mul16);
BENCHMARK(raw_mul16);

BENCHMARK_MAIN();
