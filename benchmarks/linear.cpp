#include <benchmark/benchmark.h>
#include <anna/log.hpp>
#include <Eigen/Core>

#include <anna/op.hpp>

using namespace anna::op;

template<int Channels>
static inline void run_linear10(benchmark::State & state)
{
  linear1<float, Channels, 64, 
  linear1<float, Channels, 64, 
  linear1<float, Channels, 64, 
  linear1<float, Channels, 64, 
  linear1<float, Channels, 64, 
  linear1<float, Channels, 64, 
  linear1<float, Channels, 64, 
  linear1<float, Channels, 64, 
  linear1<float, Channels, 64, 
  linear1<float, Channels, 64, 
  output<float, Channels, 64
  >>>>>>>>>>> net;

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

template<int Channels>
static inline void run_linear3(benchmark::State & state)
{
  linear1<float, Channels, 64, 
  linear1<float, Channels, 64, 
  linear1<float, Channels, 64, 
  output<float, Channels, 64
  >>>> net;

  Eigen::Matrix<float, Channels, 64> input = Eigen::Matrix<float, Channels, 64>::Zero();

  for (auto _ : state)
  {
    for (int index = 0; index < 750; ++index)
    {
      process(net, input, 64);
    }
  }
}
typedef int channels;

BENCHMARK(run_linear3<(channels)16>);
BENCHMARK(run_linear10<(channels)16>);

BENCHMARK_MAIN();
