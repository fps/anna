#include <benchmark/benchmark.h>
#include <anna/conv1d.hpp>
#include <Eigen/Core>

template<typename T, int OutputChannels, int InputChannels, int MaxBlockSize, typename NextOpType>
struct linear_op
{
  Eigen::Matrix<T, InputChannels, MaxBlockSize> m_input;
  Eigen::Matrix<T, OutputChannels, InputChannels> m_matrix;
  static int m_input_head = MaxBlockSize;

  NextOpType & m_next_op;

  inline void process(const int n)
  {
    m_next_op.m_input.middleCols(m_next_op.m_input_head, n) = m_matrix * m_input.middleCols(m_input_head, n);
    m_next_op.process(n);
  }
};

template<typename T, int InputChannels, int MaxBlockSize>
struct output
{
  Eigen::Matrix<T, InputChannels, MaxBlockSize> m_input;
  static const int m_input_head = MaxBlockSize;

  inline void process(const int n)
  {
    
  }
};

static inline void run(benchmark::State & state)
{
  linear_op<float, 4, 8, 64, 
    linear_op<float, 8, 4, 64, 
      output<float, 8, 64>
    >
  > net;
}

BENCHMARK(run);

BENCHMARK_MAIN();
