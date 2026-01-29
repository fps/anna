#include <benchmark/benchmark.h>
#include <anna/benchmark.hpp>
#include <vectorclass.h>
#include <iostream>

template<int Rows, int Cols, int N>
void mul(benchmark::State & state)
{
  float __attribute__((aligned(32))) r[Rows * N];
  float __attribute__((aligned(32))) m[Rows * Cols];
  float __attribute__((aligned(32))) v[Cols * N];

  std::fill(r, r + Rows * N, 0.0f);
  std::fill(m, m + Rows * Cols, 1.0f);
  std::fill(v, v + Cols * N, 1.0f);

  // std::cout << m[1] << "\n";

  Vec8f v_zero = 0; 
  Vec8f acc;
  Vec8f v1;
  Vec8f v2;

  const size_t inc = 8;

  for (auto _ : state)
  {
    for (int p = 0; p < 100; ++p)
    {
      /*
      escape(r);
      escape(m);
      escape(v);
      */
      for (size_t n = 0; n < N; ++n)
      {
        for (size_t row = 0; row < Rows; row += inc)
        {
          v_zero.store(r + n * Rows + row);
        }
      }
  
      for (size_t n = 0; n < N; ++n)
      {
        for (size_t row = 0; row < Rows; ++row)
        {
          /*
          v1.load(m + row * Cols + 0);
          v2.load(v + n * Cols + 0);
          acc = v1 * v2;
          */
          acc = 0;
          for (size_t col = 0; col < Cols; col += inc)
          {
            v1.load(m + row * Cols + col);
            v2.load(v + n * Cols + col);
            acc += v1 * v2;
          }
  
          *(r + n * Rows + row) = horizontal_add(acc);
        }
      }
      escape(r);
    }
  }
  // std::cout << *r << "\n";
}

#include "mul.cpp"
