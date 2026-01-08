#include <benchmark/benchmark.h>
#include <anna/benchmark.hpp>
#include <vectorclass.h>
#include <iostream>

template<int rows, int cols, int n>
void mul(benchmark::State & state)
{
  float __attribute__((aligned(32))) r[rows * n];
  float __attribute__((aligned(32))) m[rows * cols];
  float __attribute__((aligned(32))) v[cols * n];

  std::fill(r, r + rows * n, 0.0f);
  std::fill(m, m + rows * cols, 1.0f);
  std::fill(v, v + cols * n, 1.0f);

  // std::cout << m[1] << "\n";

  Vec8f vr = 0; 
  Vec8f acc;
  Vec8f v1;
  Vec8f v2;

  for (auto _ : state)
  {
    escape(r);
    escape(m);
    escape(v);

    for (size_t k = 0; k < n; ++k)
    {
      for (size_t row = 0; row < rows; row += 8)
      {
        vr.store(r + k * rows + row);
      }
    }

    for (size_t k = 0; k < n; ++k)
    {
      for (size_t row = 0; row < rows; ++row)
      {
        acc = 0;
        for (size_t col = 0; col < cols; col += 8)
        {
          v1.load(m + rows * col + col);
          v2.load(v + k * rows + col);
          acc += v1 * v2;
        }

        *(r + k * rows + row) = horizontal_add(acc);
      }
    }
    escape(r);
  }
  // std::cout << *r << "\n";
}

#include "mul.cpp"
