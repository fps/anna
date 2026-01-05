#include <benchmark/benchmark.h>
#include <anna/benchmark.hpp>
#include <blitz/array.h>
// #include <blitz/tinyvector.h>
#include <iostream>

template<int rows, int cols, int n>
void mul(benchmark::State & state)
{
  blitz::TinyMatrix<float, rows, n> r;
  blitz::TinyMatrix<float, rows, cols> m;
  blitz::TinyMatrix<float, cols, n> v;

  for (int row = 0; row < rows; ++row)
  {
    for (int col = 0; col < cols; ++col)
    {
      m(row, col) = 1;
    }

    for (int k = 0; k < n; ++k)
    {
      r(row, k) = 0;
    }
  }

  for (int col = 0; col < cols; ++col)
  {
    for (int k = 0; k < n; ++k)
    {
      v(col, k) = 0;
    }
  } 

  blitz::firstIndex i;
  blitz::secondIndex j;
  blitz::thirdIndex k;

  for (auto _ : state)
  {
    r = blitz::sum(m(i,k) * v(k,j), k);

    escape(&r);
  }
}

#include "mul.cpp"
