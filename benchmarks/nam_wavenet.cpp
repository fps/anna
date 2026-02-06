#include <benchmark/benchmark.h>

#include <anna/conv1d.hpp>
#include <anna/make_matrix_array.hpp>

#include "../examples/nam_wavenet.hpp"

template<int MaxBlockSize>
void run(benchmark::State & state)
{
  // const int max_block_size = 1 << 13;
  // const int max_block_size = 1024;
  Eigen::Matrix<float, 1, 64> input = Eigen::Matrix<float, 1, 64>::Ones();
  Eigen::Matrix<float, 1, 64> output = Eigen::Matrix<float, 1, 64>::Zero();
  auto *nam_wavenet = new anna::examples::nam_wavenet<float, 1, 1, 16, 3, 8, 3, MaxBlockSize>();

  for (auto _ : state)
  {
    for (int index = 0; index < 4096; ++index)
    {
      nam_wavenet->process(input, output, 64);
    }
  }

  delete nam_wavenet;
}

BENCHMARK(run<64 << 0>);
BENCHMARK(run<64 << 1>);
BENCHMARK(run<64 << 5>);

BENCHMARK_MAIN();
