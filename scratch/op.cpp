#include <benchmark/benchmark.h>
#include <anna/conv1d.hpp>
#include <anna/log.hpp>
#include <Eigen/Core>

template<typename T, int Channels, int MaxBlockSize, typename NextOpType>
struct linear1
{
  Eigen::Matrix<T, Channels, MaxBlockSize> m_input;
  Eigen::Matrix<T, Channels, Channels> m_matrix;
  static const int m_input_head = 0;

  NextOpType m_next_op;

  linear1() :
    m_matrix(Eigen::Matrix<T, Channels, Channels>::Zero())
  {
    for (int index = 0; index < Channels; ++index)
    {
      m_matrix(index, index) = 1;
    }
  }

  template<int n, typename ValueType>
  inline void set(ValueType value)
  {
    if constexpr (0 == n)
    {
      m_matrix = value;
    }
    else
    {
      m_next_op.set(n - 1, value);
    }
  }  

  inline auto & end() { return m_next_op.end(); }

  inline auto & input() { return m_next_op.input(); }

  inline int input_head() { return m_next_op.input_head(); }

  inline void process(const int n)
  {
    input().middleCols(m_next_op.input_head(), n) = m_matrix * input().middleCols(m_input_head, n);
    m_next_op.process(n);
  }
};

template<typename T, int OutputChannels, int InputChannels, int MaxBlockSize, typename NextOpType>
struct linear2
{
  Eigen::Matrix<T, InputChannels, MaxBlockSize> m_input;
  Eigen::Matrix<T, OutputChannels, InputChannels> m_matrix;
  static const int m_input_head = 0;

  NextOpType m_next_op;

  linear2() :
    m_matrix(Eigen::Matrix<T, OutputChannels, InputChannels>::Zero())
  {
    for (int index = 0; index < std::min(InputChannels, OutputChannels); ++index)
    {
      m_matrix(index, index) = 1;
    }
  }

  template<int n, typename ValueType>
  inline void set(ValueType value)
  {
    if constexpr (0 == n)
    {
      m_matrix = value;
    }
    else
    {
      m_next_op.set(n - 1, value);
    }
  }  

  inline auto & end() { return m_next_op.end(); }

  inline auto & input() { return m_input; }

  inline int input_head() { return m_input_head; }

  inline void process(const int n)
  {
    m_next_op.input().middleCols(m_next_op.input_head(), n).noalias() = m_matrix * m_input.middleCols(m_input_head, n);
    m_next_op.process(n);
  }
};


template<typename T, int Channels, typename NextOpType>
struct vector_add
{
  Eigen::Vector<T, Channels> m_vector;

  NextOpType m_next_op;

  template<int n, typename ValueType>
  inline void set(ValueType value)
  {
    if constexpr (0 == n)
    {
      m_vector = value;
    }
    else
    {
      m_next_op.set(n - 1, value);
    }
  }  

  inline auto & end() { return m_next_op.end(); }

  inline auto & input() { return m_next_op.input(); }

  inline int input_head() { return m_next_op.input_head(); }

  inline void process(const int n)
  {
    m_next_op.input().middleCols(m_next_op.input_head(), n).colwise().noalias() += m_vector;
    m_next_op.process(n);
  }
};

template<typename T, int InputChannels, int MaxBlockSize>
struct output
{
  Eigen::Matrix<T, InputChannels, MaxBlockSize> m_input;
  static const int m_input_head = 0;

  template<int n, typename ValueType>
  inline void set(ValueType value)
  {
    if constexpr (0 == n)
    {
      ERR("output has no value to set()")
    }
  }  

  inline auto & end() { return *this; }

  inline auto & input() { return m_input; }

  inline int input_head() { return m_input_head; }

  inline void process(const int n)
  {
    // NO OP 
  }
};

template<typename T, int Nominator, int Denominator, typename NextOpType>
struct scalar_multiple
{
  NextOpType m_next_op;

  T m_value = (T)Nominator/(T)Denominator;

  scalar_multiple() :
    m_value((T)Nominator/(T)Denominator)
  {

  }

  template<int n, typename ValueType>
  inline void set(ValueType value)
  {
    if constexpr (0 == n)
    {
      m_value = value;
    }
    else
    {
      m_next_op.set(n - 1, value);
    }
  }  

  inline auto & end() { return m_next_op.end(); }

  inline auto & input() { return m_next_op.input(); }

  inline int input_head() { return m_next_op.input_head(); }

  inline void process(const int n)
  {
    m_next_op.input().middleCols(m_next_op.input_head(), n).array() *= m_value;
    m_next_op.process(n);
  }
};

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
