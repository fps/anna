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

static void eigen_mul64(benchmark::State & state)
{
  Eigen::Matrix<float, 64, 1> r = Eigen::Matrix<float, 64, 1>::Zero();
  Eigen::Matrix<float, 64, 64> m = Eigen::Matrix<float, 64, 64>::Ones();
  Eigen::Matrix<float, 64, 1> v = Eigen::Matrix<float, 64, 1>::Ones();
  
  for (auto _ : state)
  {
    r = m * v;

    escape(&r);
  }
}


static void raw_mul64(benchmark::State & state)
{
  float r[64];
  float m[64*64];
  float v[64];

  for (auto _ : state)
  {
    for (size_t row = 0; row < 64; ++row)
    {
      r[row] = 0;
      for (size_t col = 0; col < 64; ++col)
      {
        r[row] += m[col*64+row] * v[row];
      }
    }
    escape(r);
  }
}

static void eigen_mul32(benchmark::State & state)
{
  Eigen::Matrix<float, 32, 1> r = Eigen::Matrix<float, 32, 1>::Zero();
  Eigen::Matrix<float, 32, 32> m = Eigen::Matrix<float, 32, 32>::Ones();
  Eigen::Matrix<float, 32, 1> v = Eigen::Matrix<float, 32, 1>::Ones();
  
  for (auto _ : state)
  {
    r = m * v;

    escape(&r);
  }
}

static void raw_mul32(benchmark::State & state)
{
  float r[32];
  float m[32*32];
  float v[32];

  for (auto _ : state)
  {
    for (size_t row = 0; row < 32; ++row)
    {
      r[row] = 0;
      for (size_t col = 0; col < 32; ++col)
      {
        r[row] += m[col*32+row] * v[row];
      }
    }
    escape(r);
  }
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


BENCHMARK(eigen_mul64);
BENCHMARK(eigen_mul32);
BENCHMARK(eigen_mul16);

BENCHMARK(raw_mul64);
BENCHMARK(raw_mul32);
BENCHMARK(raw_mul16);

BENCHMARK_MAIN();
