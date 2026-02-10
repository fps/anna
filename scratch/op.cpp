#include <benchmark/benchmark.h>
#include <anna/conv1d.hpp>
#include <anna/log.hpp>
#include <Eigen/Core>

#include <scratch/op.hpp>

using namespace anna::op;

template<int Channels>
static inline void run(benchmark::State & state)
{
  linear1<float, Channels, 64, 
  linear1<float, Channels, 64, 
  linear1<float, Channels, 64, 
  linear1<float, Channels, 64, 
  linear1<float, Channels, 64, 
  scalar_multiple<float, 17, 3, 
  scalar_multiple<float, 1, 2, 
  linear1<float, Channels, 64, 
  linear1<float, Channels, 64, 
  linear1<float, Channels, 64, 
  linear1<float, Channels, 64, 
  linear1<float, Channels, 64, 
  output<float, Channels, 64
  >>>>>>>>>>>>> net;

  net.m_input = Eigen::Matrix<float, Channels, 64>::Ones();

  net.template set<0>(Eigen::Matrix<float, Channels, Channels>::Zero());

  for (auto _ : state)
  {
    for (int index = 0; index < 750; ++index)
    {
      net.process(64);
    }
  }

  // std::cout << net.end().m_input;
}

typedef int channels;

BENCHMARK(run<(channels)8>);
BENCHMARK(run<(channels)16>);
BENCHMARK(run<(channels)32>);

BENCHMARK_MAIN();
