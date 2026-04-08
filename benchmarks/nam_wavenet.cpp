#include <benchmark/benchmark.h>

#include <anna/conv1d.hpp>
#include <anna/make_matrix_array.hpp>

#include "../examples/nam_wavenet.hpp"
#include <sys/mman.h>

#define NBLOCKS 750

void run(benchmark::State & state)
{
  Eigen::Matrix<float, 1, 64 * NBLOCKS> input = Eigen::Matrix<float, 1, 64 * NBLOCKS>::Ones();
  Eigen::Matrix<float, 1, 64 * NBLOCKS> output = Eigen::Matrix<float, 1, 64 * NBLOCKS>::Zero();
  auto *nam_wavenet = new anna::examples::nam_wavenet<float, 1, 1, 16, 3, 8, 3, 64>();
  mlockall(MCL_CURRENT);

  for (auto _ : state)
  {
    for (int index = 0; index < NBLOCKS; ++index)
    {
      nam_wavenet->process(input.middleCols(index*64, 64), output.middleCols(index*64, 64), 64);
    }
  }

  delete nam_wavenet;
}

BENCHMARK(run);

BENCHMARK_MAIN();
