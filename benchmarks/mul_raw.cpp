#include <benchmark/benchmark.h>
#include <anna/benchmark.hpp>

template<int rows, int cols, int n>
void mul(benchmark::State & state)
{
  float __attribute__((aligned(16))) r[rows * n];
  float __attribute__((aligned(16))) m[rows * cols];
  float __attribute__((aligned(16))) v[cols * n];

  for (auto _ : state)
  {
    for (size_t k = 0; k < n; ++k)
    {
      for (size_t row = 0; row < rows; ++row)
      {
        r[k * rows + row] = 0;
      }
    }

    for (size_t k = 0; k < n; ++k)
    {
      for (size_t row = 0; row < rows; ++row)
      {
        for (size_t col = 0; col < cols; ++col)
        {
          r[k * rows + row] += m[col * rows + row] * v[col + k * cols];
        }
      }
    }
    escape(r);
  }
}

#include "mul.cpp"
