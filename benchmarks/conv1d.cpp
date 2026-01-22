#include <Eigen/Core>
#include <benchmark/benchmark.h>
#include <anna/conv1d_double_buffer.hpp>
#include <anna/benchmark.hpp>

template<typename T, int Channels, int k>
std::array<Eigen::Matrix<T, Channels, Channels>, k>
make_weights()
{
  std::array<Eigen::Matrix<T, Channels, Channels>, k> weights;
  weights.fill(Eigen::Matrix<T, Channels, Channels>::Ones());
  return weights;
}

template<typename T, int KernelSize, int Channels, int Dilation>
struct conv1d_1
{
  const std::array<Eigen::Matrix<T, Channels, Channels>, KernelSize> m_weights;
  const Eigen::Vector<T, Channels> m_bias;
  const Eigen::Matrix<T, Channels, Channels> m_expanded_bias;

  conv1d_1() :
    m_weights(make_weights<T, Channels, KernelSize>()),
    m_bias(Eigen::Vector<T, Channels>::Ones()),
    m_expanded_bias(Eigen::Matrix<T, Channels, Channels>::Ones())
  {

  }

  template<typename T1, typename T2>
  inline void process1(Eigen::MatrixBase<T1> const & input, Eigen::MatrixBase<T2> const & output, const int & head, const int & n) const
  {
    const_cast<Eigen::MatrixBase<T2> &>(output).template middleCols(head - n, n).noalias() = std::get<0>(m_weights) * input.template middleCols(head - ((KernelSize-1) * Dilation + n), n);
    for (int k = (KernelSize-2); k >= 0; --k)
    {
      const_cast<Eigen::MatrixBase<T2> &>(output).template middleCols(head - n, n) += m_weights[k] * input.template middleCols(head - ((k * Dilation)) - n, n);
    }
    
    const_cast<Eigen::MatrixBase<T2> &>(output).template middleCols(head - n, n).colwise() += m_bias;
  }
};


template<typename T, int KernelSize, int Channels, int Dilation>
struct conv1d_2
{
  const std::array<Eigen::Matrix<T, Channels, Channels>, KernelSize> m_weights;
  const Eigen::Vector<T, Channels> m_bias;
  const Eigen::Matrix<T, Channels, Channels> m_expanded_bias;

  conv1d_2() :
    m_weights(make_weights<T, Channels, KernelSize>()),
    m_bias(Eigen::Vector<T, Channels>::Ones()),
    m_expanded_bias(Eigen::Matrix<T, Channels, Channels>::Ones())
  {

  }

  template<typename T1, typename T2>
  inline void process2(Eigen::MatrixBase<T1> const & input, Eigen::MatrixBase<T2> const & output, const int head, const int n) const 
  {
    const_cast<Eigen::MatrixBase<T2> &>(output).template middleCols(head - n, n).noalias() =
        std::get<0>(m_weights) * input.template middleCols(head - ((KernelSize-1) * Dilation + n), n)
      + std::get<1>(m_weights) * input.template middleCols(head - ((KernelSize-2) * Dilation + n), n)
      + std::get<2>(m_weights) * input.template middleCols(head - ((KernelSize-3) * Dilation + n), n);
    
    const_cast<Eigen::MatrixBase<T2> &>(output).template middleCols(head - n, n).colwise() += m_bias;
  }
};

template<typename T, int KernelSize, int Channels, int Dilation>
struct conv1d_3
{
  const std::array<Eigen::Matrix<T, Channels, Channels>, KernelSize> m_weights;
  const Eigen::Vector<T, Channels> m_bias;
  const Eigen::Matrix<T, Channels, Channels> m_expanded_bias;

  conv1d_3() :
    m_weights(make_weights<T, Channels, KernelSize>()),
    m_bias(Eigen::Vector<T, Channels>::Ones()),
    m_expanded_bias(Eigen::Matrix<T, Channels, Channels>::Ones())
  {

  }

  template<typename T1, typename T2>
  inline void process3(Eigen::MatrixBase<T1> const & input, Eigen::MatrixBase<T2> const & output, const int & head, const int & n) const 
  {
    const_cast<Eigen::MatrixBase<T2> &>(output).template middleCols(head - n, n).noalias() = 
        std::get<0>(m_weights) * input.template middleCols(head - ((KernelSize-1) * Dilation + n), n)
      + std::get<1>(m_weights) * input.template middleCols(head - ((KernelSize-2) * Dilation + n), n)
      + std::get<2>(m_weights) * input.template middleCols(head - ((KernelSize-3) * Dilation + n), n)
      + m_expanded_bias;
  }
};

static void one(benchmark::State & state)
{
  const conv1d_1<float, 3, 16, 1024> c;
  const Eigen::Matrix<float, 16, 4096> input = Eigen::Matrix<float, 16, 4096>::Ones();
  Eigen::Matrix<float, 16, 4096> output;

  for (auto _ : state)
  {
    c.process1(input, output, 4096, 64);
    escape(&output);
  }
}

static void two(benchmark::State & state)
{
  const conv1d_2<float, 3, 16, 1024> c;
  const Eigen::Matrix<float, 16, 4096> input = Eigen::Matrix<float, 16, 4096>::Ones();
  Eigen::Matrix<float, 16, 4096> output;

  for (auto _ : state)
  {
    c.process2(input, output, 4096, 64);
    escape(&output);
  }
}

static void three(benchmark::State & state)
{
  const conv1d_3<float, 3, 16, 1024> c;
  const Eigen::Matrix<float, 16, 4096> input = Eigen::Matrix<float, 16, 4096>::Ones();
  Eigen::Matrix<float, 16, 4096> output;

  for (auto _ : state)
  {
    c.process3(input, output, 4096, 64);
    escape(&output);
  }
}

static void standard(benchmark::State & state)
{
  anna::conv1d_double_buffer<float, 64, 3, 16, 16, true, 1024> conv1d;
  Eigen::Matrix<float, 16, 64> input = Eigen::Matrix<float, 16, 64>::Ones();
  Eigen::Matrix<float, 16, 64> output;

  for (auto _ : state)
  {
    conv1d.process(input, output, 64);
    escape(&output);
  }
}

BENCHMARK(one);
BENCHMARK(two);
BENCHMARK(three);
BENCHMARK(standard);

BENCHMARK_MAIN();
