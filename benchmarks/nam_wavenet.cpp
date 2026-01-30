#include <benchmark/benchmark.h>

#include <anna/conv1d.hpp>
#include <anna/make_matrix_array.hpp>

#include "../examples/nam_wavenet.hpp"

void run(benchmark::State & state)
{
  Eigen::Matrix<float, 1, 64> input = Eigen::Matrix<float, 1, 64>::Ones();
  Eigen::Matrix<float, 1, 64> output = Eigen::Matrix<float, 1, 64>::Zero();
  anna::examples::nam_wavenet<float, 1, 1, 4096> *nam_wavenet = new anna::examples::nam_wavenet<float, 1, 1, 4096>();

  for (auto _ : state)
  {
    for (int index = 0; index < 750; ++index)
    {
      nam_wavenet->process(input, output, 64);
    }
  }

  delete nam_wavenet;
}

BENCHMARK(run);

BENCHMARK_MAIN();
