#include <benchmark/benchmark.h>
#include <anna/benchmark.hpp>
#include <blitz/array.h>

template<int rows, int cols, int n>
void mul_blitz(benchmark::State & state)
{
  blitz::TinyMatrix<float, rows, n> r;
  blitz::TinyMatrix<float, rows, cols> m;
  blitz::TinyMatrix<float, rows, n> v;
  /*
  Eigen::Matrix<float, rows, n> r = Eigen::Matrix<float, rows, n>::Zero();
  Eigen::Matrix<float, rows, cols> m = Eigen::Matrix<float, rows, cols>::Ones();
  Eigen::Matrix<float, rows, n> v = Eigen::Matrix<float, cols, n>::Ones();
  */ 
  for (auto _ : state)
  {
    r = m * v;

    escape(&r);
  }
}


BENCHMARK(mul_blitz<64, 64, 1024>);
BENCHMARK(mul_blitz<32, 32, 1024>);
BENCHMARK(mul_blitz<16, 16, 1024>);
BENCHMARK(mul_blitz<8, 8, 1024>);

BENCHMARK(mul_blitz<64, 64, 128>);
BENCHMARK(mul_blitz<32, 32, 128>);
BENCHMARK(mul_blitz<16, 16, 128>);
BENCHMARK(mul_blitz<8, 8, 128>);

BENCHMARK(mul_blitz<64, 64, 64>);
BENCHMARK(mul_blitz<32, 32, 64>);
BENCHMARK(mul_blitz<16, 16, 64>);
BENCHMARK(mul_blitz<8, 8, 64>);

BENCHMARK(mul_blitz<64, 64, 1>);
BENCHMARK(mul_blitz<32, 32, 1>);
BENCHMARK(mul_blitz<16, 16, 1>);
BENCHMARK(mul_blitz<8, 8, 1>);

BENCHMARK_MAIN();
