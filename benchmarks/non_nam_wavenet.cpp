#include <anna/anna.hpp>
#include <benchmark/benchmark.h>
#include <anna/benchmark.hpp>

#include <sys/resource.h>

template<int nchannels, int buffer_size, int process_size, int n>
void run(benchmark::State & state)
{
  /*
  struct rlimit rl;
  getrlimit(RLIMIT_STACK, &rl);
  rl.rlim_cur = 128 * 1024 * 1024;
  setrlimit(RLIMIT_STACK, &rl);
  */

	auto model = new anna::sequence::input_output::model<
      anna::conv1d<float, buffer_size, 1, 1, nchannels, true, 1>,
      anna::conv1d<float, buffer_size, 3, nchannels, nchannels, true, 1>,
      anna::conv1d<float, buffer_size, 3, nchannels, nchannels, true, 2>,
      anna::conv1d<float, buffer_size, 3, nchannels, nchannels, true, 4>,
      anna::conv1d<float, buffer_size, 3, nchannels, nchannels, true, 8>,
      anna::conv1d<float, buffer_size, 3, nchannels, nchannels, true, 16>,
      anna::conv1d<float, buffer_size, 3, nchannels, nchannels, true, 32>,
      anna::conv1d<float, buffer_size, 3, nchannels, nchannels, true, 64>,
      anna::conv1d<float, buffer_size, 3, nchannels, nchannels, true, 128>,
      anna::conv1d<float, buffer_size, 3, nchannels, nchannels, true, 256>,
      anna::conv1d<float, buffer_size, 3, nchannels, nchannels, true, 512>,
      anna::conv1d<float, buffer_size, 3, nchannels, nchannels, true, 1>,
      anna::conv1d<float, buffer_size, 3, nchannels, nchannels, true, 2>,
      anna::conv1d<float, buffer_size, 3, nchannels, nchannels, true, 4>,
      anna::conv1d<float, buffer_size, 3, nchannels, nchannels, true, 8>,
      anna::conv1d<float, buffer_size, 3, nchannels, nchannels, true, 16>,
      anna::conv1d<float, buffer_size, 3, nchannels, nchannels, true, 32>,
      anna::conv1d<float, buffer_size, 3, nchannels, nchannels, true, 64>,
      anna::conv1d<float, buffer_size, 3, nchannels, nchannels, true, 128>,
      anna::conv1d<float, buffer_size, 3, nchannels, nchannels, true, 256>,
      anna::conv1d<float, buffer_size, 3, nchannels, nchannels, true, 512>,
      anna::conv1d<float, buffer_size, 1, nchannels, 1, true, 512>
    >();

  Eigen::Matrix<float, 1, n> input;

  for (auto _ : state)
  {
    model->process(input.template leftCols(process_size), process_size);
  }

  delete model;
}

BENCHMARK(run<64, 128, 64, 64>);
BENCHMARK(run<32, 128, 64, 64>);
BENCHMARK(run<16, 128, 64, 64>);
BENCHMARK(run<8, 128, 64, 64>);

BENCHMARK_MAIN();
