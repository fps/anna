#include <benchmark/benchmark.h>
#include <anna/benchmark.hpp>

template<int rows, int cols, int n>
void raw(benchmark::State & state)
{
  float r[rows];
  float m[rows*cols];
  float v[cols];

  for (auto _ : state)
  {
    for (size_t row = 0; row < rows; ++row)
    {
      r[row] = 0;
      for (size_t col = 0; col < cols; ++col)
      {
        r[row] += m[col*cols+row] * v[row];
      }
    }
    escape(r);
  }
}

BENCHMARK(raw<64, 64, 1>);
BENCHMARK(raw<32, 32, 1>);
BENCHMARK(raw<16, 16, 1>);
BENCHMARK(raw<8, 8, 1>);

BENCHMARK_MAIN();
