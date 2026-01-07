#include <vectorclass.h>
#include <benchmark/benchmark.h>
#include <anna/benchmark.hpp>

template<int n>
void dot_raw(benchmark::State & state)
{
  auto v1 = new float[n];
  auto v2 = new float[n];

  for (auto _ : state)
  {
    float r = 0;
    for (int k = 0; k < n; ++k)
    {
      r += v1[k] * v2[k];
    }
    escape(&r);
  }
}

template<int n>
void dot_vector(benchmark::State & state)
{
  auto v1 = new float[n];
  auto v2 = new float[n];

  for (auto _ : state)
  {
    float r = 0;
    Vec8f vv1;
    Vec8f vv2;

    for (int k = 0; k < n; k += 8)
    {
      vv1.load(v1 + k);
      vv2.load(v2 + k);

      r += horizontal_add(vv1 * vv2);
    }
    escape(&r);
  }
}

BENCHMARK(dot_raw<1024*512>);
BENCHMARK(dot_vector<1024*512>);

BENCHMARK_MAIN();
