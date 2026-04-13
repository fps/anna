#include <benchmark/benchmark.h>

#include <anna/conv1d.hpp>
#include <anna/make_matrix_array.hpp>

#include "../examples/nam_wavenet.hpp"
#include <sys/mman.h>

#define NBLOCKS 4096
// #define NBLOCKS 750


template<int BlockSize, int NumberOfBlocks>
void run(benchmark::State & state)
{
  std::vector<float> input(BlockSize * NumberOfBlocks, 1.0f);
  std::vector<float> output(BlockSize * NumberOfBlocks, 0.0f);

  auto *nam_wavenet = new anna::examples::nam_wavenet<float, 1, 1, 16, 3, 8, 3, BlockSize>();
  mlockall(MCL_CURRENT);

  for (auto _ : state)
  {
    for (int index = 0; index < NumberOfBlocks; ++index)
    {
      Eigen::Map<Eigen::Matrix<float, 1, BlockSize>> map_in(input.data() + BlockSize * index);
      Eigen::Map<Eigen::Matrix<float, 1, BlockSize>> map_out(output.data() + BlockSize * index);

      nam_wavenet->process(map_in, map_out, BlockSize);
    }
  }

  delete nam_wavenet;
}

BENCHMARK(run<64, 750>);
BENCHMARK(run<64, 4096>);

BENCHMARK_MAIN();
