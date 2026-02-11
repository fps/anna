#include <benchmark/benchmark.h>
#include <anna/op.hpp>

static inline void run(benchmark::State & state)
{
  std::tuple<
    anna::linear<float, 16, 16, 64>,
    anna::linear<float, 16, 16, 64>,
    anna::linear<float, 16, 16, 64>,
    anna::linear<float, 16, 16, 64>
  > net;

  for (auto _ : state)
  {

  }
}

BENCHMARK(run);
BENCHMARK_MAIN();
