#include <vectorclass.h>
#include <benchmark/benchmark.h>
#include <anna/benchmark.hpp>
#include <iostream>

template<int n>
void dot_raw(benchmark::State & state)
{
  auto v1 = new float[n];
  auto v2 = new float[n];

  for (int k = 0; k < n; ++k)
  {
    v1[k] = 1.0f;
    v2[k] = 1.0f;
  }

  float r;

  for (auto _ : state)
  {
    r = 0;
    for (int k = 0; k < n; ++k)
    {
      r += v1[k] * v2[k];
    }
    escape(&r);
  }

  delete[] v1;
  delete[] v2;

  // std::cout << n << ": " << r << "\n";
}

template<int n>
void dot_vector(benchmark::State & state)
{
  auto v1 = new float[n];
  auto v2 = new float[n];

  for (int k = 0; k < n; ++k)
  {
    v1[k] = 1.0f;
    v2[k] = 1.0f;
  }

  float r;
  Vec8f vv1;
  Vec8f vv2;
  Vec8f im;

  for (auto _ : state)
  {
    r = 0;

    im = 0;
    for (int k = 0; k < n; k += 8)
    {
      vv1.load(v1 + k);
      vv2.load(v2 + k);

      im += vv1 * vv2;
    }

    r = horizontal_add(im);

    escape(&r);
  }

  delete[] v1;
  delete[] v2;

  // std::cout << n << ": " << r << "\n";
}

BENCHMARK(dot_raw<1024*512>);
BENCHMARK(dot_raw<1024*32>);
BENCHMARK(dot_raw<1024>);
BENCHMARK(dot_raw<512>);
BENCHMARK(dot_raw<128>);
BENCHMARK(dot_raw<64>);
BENCHMARK(dot_raw<32>);

BENCHMARK(dot_vector<1024*512>);
BENCHMARK(dot_vector<1024*32>);
BENCHMARK(dot_vector<1024>);
BENCHMARK(dot_vector<512>);
BENCHMARK(dot_vector<128>);
BENCHMARK(dot_vector<64>);
BENCHMARK(dot_vector<32>);

BENCHMARK_MAIN();
