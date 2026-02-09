#include <benchmark/benchmark.h>
#include <anna/conv1d.hpp>
#include <Eigen/Core>

template<typename T, int OutputChannels, int InputChannels, int MaxBlockSize, typename NextOpType>
struct linear
{
  Eigen::Matrix<T, InputChannels, MaxBlockSize> m_input;
  Eigen::Matrix<T, OutputChannels, InputChannels> m_matrix;
  static const int m_input_head = MaxBlockSize;

  NextOpType m_next_op;

  linear() :
    m_matrix(Eigen::Matrix<T, OutputChannels, InputChannels>::Zero())
  {
    for (int index = 0; index < std::min(InputChannels, OutputChannels); ++index)
    {
      m_matrix(index, index) = 1;
    }
  }

  auto & end() { return m_next_op.end(); }

  auto & input() { return m_input; }

  int input_head() { return m_input_head; }

  inline void process(const int n)
  {
    m_next_op.input().middleCols(m_next_op.input_head() - n, n).noalias() = m_matrix * m_input.middleCols(m_input_head - n, n);

    m_next_op.process(n);
  }
};

template<typename T, int Channels, typename NextOpType>
struct vector_add
{
  Eigen::Vector<T, Channels> m_value;

  NextOpType m_next_op;

  auto & end() { return m_next_op.end(); }

  auto & input() { return m_next_op.input(); }

  int input_head() { return m_next_op.input_head(); }

  inline void process(const int n)
  {
    m_next_op.input().middleCols(m_next_op.input_head() - n, n).colwise().noalias() += m_value;
    m_next_op.process(n);
  }
};

template<typename T, int InputChannels, int MaxBlockSize>
struct output
{
  Eigen::Matrix<T, InputChannels, MaxBlockSize> m_input;
  static const int m_input_head = MaxBlockSize;

  auto & end() { return *this; }

  auto & input() { return m_input; }

  int input_head() { return m_input_head; }

  inline void process(const int n)
  {
    // NO OP 
  }
};

template<typename T, int Nominator, int Denominator, typename NextOpType>
struct scalar_multiple
{
  NextOpType m_next_op;

  static constexpr T m_value = (T)Nominator/(T)Denominator;

  auto & end() { return m_next_op.end(); }

  auto & input() { return m_next_op.input(); }

  int input_head() { return m_next_op.input_head(); }

  inline void process(const int n)
  {
    m_next_op.input().middleCols(m_next_op.input_head() - n, n).array() *= m_value;
    m_next_op.process(n);
  }
};

template<int Channels>
static inline void run(benchmark::State & state)
{
  linear<float, Channels, Channels, 64, 
  linear<float, Channels, Channels, 64, 
  linear<float, Channels, Channels, 64, 
  linear<float, Channels, Channels, 64, 
  linear<float, Channels, Channels, 64, 
  scalar_multiple<float, 17, 3, 
  linear<float, Channels, Channels, 64, 
  linear<float, Channels, Channels, 64, 
  linear<float, Channels, Channels, 64, 
  linear<float, Channels, Channels, 64, 
  linear<float, Channels, Channels, 64, 
  output<float, Channels, 64
  >>>>>>>>>>>> net;

  net.m_input = Eigen::Matrix<float, Channels, 64>::Ones();

  for (auto _ : state)
  {
    for (int index = 0; index < 750; ++index)
    {
      net.process(64);
    }
  }

  // std::cout << net.end().m_input;
}

BENCHMARK(run<8>);
BENCHMARK(run<16>);
BENCHMARK(run<32>);

BENCHMARK_MAIN();
