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
   

template<int Channels>
static inline void run_conv1d_bias(benchmark::State & state)
{
  conv1d<float, Channels, Channels, 3, 1024, 64, 
  vector_add<float, Channels, 
  output<float, Channels, 64
  >>> net;  

  Eigen::Matrix<float, Channels, 64> input = Eigen::Matrix<float, Channels, 64>::Zero();

  for (auto _ : state)
  {
    for (int index = 0; index < 750; ++index)
    {
      process(net, input, 64);
    }
  }
}

template<int Channels>
static inline void run_conv1d_bias20(benchmark::State & state)
{
  linear2<float, Channels, 1, 64,
  conv1d<float, Channels, Channels, 3, (1 << 0), 64, 
  vector_add<float, Channels, 
  anna::op::tanh<
  conv1d<float, Channels, Channels, 3, 1 << 1, 64, 
  vector_add<float, Channels, 
  anna::op::tanh<
  conv1d<float, Channels, Channels, 3, 1 << 2, 64, 
  vector_add<float, Channels, 
  anna::op::tanh<
  conv1d<float, Channels, Channels, 3, 1 << 3, 64, 
  vector_add<float, Channels, 
  anna::op::tanh<
  conv1d<float, Channels, Channels, 3, 1 << 4, 64, 
  vector_add<float, Channels, 
  anna::op::tanh<
  conv1d<float, Channels, Channels, 3, 1 << 5, 64, 
  vector_add<float, Channels, 
  anna::op::tanh<
  conv1d<float, Channels, Channels, 3, 1 << 6, 64, 
  vector_add<float, Channels, 
  anna::op::tanh<
  conv1d<float, Channels, Channels, 3, 1 << 7, 64, 
  vector_add<float, Channels, 
  anna::op::tanh<
  conv1d<float, Channels, Channels, 3, 1 << 8, 64, 
  vector_add<float, Channels, 
  anna::op::tanh<
  conv1d<float, Channels, Channels, 3, 1 << 9, 64, 
  vector_add<float, Channels, 
  anna::op::tanh<
  conv1d<float, Channels, Channels, 3, 1 << 0, 64, 
  vector_add<float, Channels, 
  anna::op::tanh<
  conv1d<float, Channels, Channels, 3, 1 << 1, 64, 
  vector_add<float, Channels, 
  anna::op::tanh<
  conv1d<float, Channels, Channels, 3, 1 << 2, 64, 
  vector_add<float, Channels, 
  anna::op::tanh<
  conv1d<float, Channels, Channels, 3, 1 << 3, 64, 
  vector_add<float, Channels, 
  anna::op::tanh<
  conv1d<float, Channels, Channels, 3, 1 << 4, 64, 
  vector_add<float, Channels, 
  anna::op::tanh<
  conv1d<float, Channels, Channels, 3, 1 << 5, 64, 
  vector_add<float, Channels, 
  anna::op::tanh<
  conv1d<float, Channels, Channels, 3, 1 << 6, 64, 
  vector_add<float, Channels, 
  anna::op::tanh<
  conv1d<float, Channels, Channels, 3, 1 << 7, 64, 
  vector_add<float, Channels, 
  anna::op::tanh<
  conv1d<float, Channels, Channels, 3, 1 << 8, 64, 
  vector_add<float, Channels, 
  anna::op::tanh<
  conv1d<float, Channels, Channels, 3, 1 << 9, 64, 
  vector_add<float, Channels, 
  anna::op::tanh<
  linear2<float, 1, Channels, 64, 
  output<float, 1, 64
  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> net;  

  Eigen::Matrix<float, 1, 64> input = Eigen::Matrix<float, 1, 64>::Zero();

  for (auto _ : state)
  {
    for (int index = 0; index < 750; ++index)
    {
      process(net, input, 64);
    }
  }
}

template<int Channels>
static inline void run_conv1d(benchmark::State & state)
{
  conv1d<float, Channels, Channels, 3, 1024, 64, 
  output<float, Channels, 64
  >> net;  

  Eigen::Matrix<float, Channels, 64> input = Eigen::Matrix<float, Channels, 64>::Zero();

  for (auto _ : state)
  {
    for (int index = 0; index < 750; ++index)
    {
      process(net, input, 64);
    }
  }
}

template<int Channels>
static inline void run_conv1d_bias_tanh20(benchmark::State & state)
{
  conv1d_bias_tanh<float, Channels, Channels, 3, (1 << 0), 64,
  conv1d_bias_tanh<float, Channels, Channels, 3, (1 << 1), 64,
  conv1d_bias_tanh<float, Channels, Channels, 3, (1 << 2), 64,
  conv1d_bias_tanh<float, Channels, Channels, 3, (1 << 3), 64,
  conv1d_bias_tanh<float, Channels, Channels, 3, (1 << 4), 64,
  conv1d_bias_tanh<float, Channels, Channels, 3, (1 << 5), 64,
  conv1d_bias_tanh<float, Channels, Channels, 3, (1 << 6), 64,
  conv1d_bias_tanh<float, Channels, Channels, 3, (1 << 7), 64,
  conv1d_bias_tanh<float, Channels, Channels, 3, (1 << 8), 64,
  conv1d_bias_tanh<float, Channels, Channels, 3, (1 << 9), 64,
  conv1d_bias_tanh<float, Channels, Channels, 3, (1 << 0), 64,
  conv1d_bias_tanh<float, Channels, Channels, 3, (1 << 1), 64,
  conv1d_bias_tanh<float, Channels, Channels, 3, (1 << 2), 64,
  conv1d_bias_tanh<float, Channels, Channels, 3, (1 << 3), 64,
  conv1d_bias_tanh<float, Channels, Channels, 3, (1 << 4), 64,
  conv1d_bias_tanh<float, Channels, Channels, 3, (1 << 5), 64,
  conv1d_bias_tanh<float, Channels, Channels, 3, (1 << 6), 64,
  conv1d_bias_tanh<float, Channels, Channels, 3, (1 << 7), 64,
  conv1d_bias_tanh<float, Channels, Channels, 3, (1 << 8), 64,
  conv1d_bias_tanh<float, Channels, Channels, 3, (1 << 9), 64,
  output<float, Channels, 64
  >>>>>>>>>>>>>>>>>>>>> net;

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

BENCHMARK(run_conv1d<(channels)16>);
BENCHMARK(run_conv1d_bias<(channels)16>);
BENCHMARK(run_conv1d_bias_tanh20<(channels)16>);
BENCHMARK(run_conv1d_bias20<(channels)16>);
BENCHMARK(run_conv1d_bias20<(channels)32>);
BENCHMARK(run_linear3<(channels)16>);
BENCHMARK(run_linear10<(channels)16>);

BENCHMARK_MAIN();
