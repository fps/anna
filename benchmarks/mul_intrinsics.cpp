#include <benchmark/benchmark.h>
#include <anna/benchmark.hpp>

typedef float v4sf __attribute__ ((vector_size (16)));

template<int rows, int cols, int n>
void mul(benchmark::State & state)
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

#include "mul.cpp"
