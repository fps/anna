#include <Eigen/Core>
#include <benchmark/benchmark.h>
#include <anna/conv1d.hpp>
#include <anna/benchmark.hpp>

static void standard(benchmark::State & state)
{
  anna::conv1d<float, 64, 3, 16, 16, true, 1024> conv1d;
  Eigen::Matrix<float, 16, 64> input = Eigen::Matrix<float, 16, 64>::Ones();
  Eigen::Matrix<float, 16, 64> output;

  for (auto _ : state)
  {
    conv1d.process(input, output, 64);
    escape(&output);
  }
}

template<typename T, int KernelSize, int Channels, int Dilation>
struct conv1d_1
{
  std::array<Eigen::Matrix<T, Channels, Channels>, KernelSize> m_weights;
  Eigen::Vector<T, Channels> m_bias;
  Eigen::Matrix<T, Channels, Channels> m_expanded_bias;

  template<typename T1, typename T2>
  inline void process1(Eigen::MatrixBase<T1> const & input, Eigen::MatrixBase<T2> const & output, const int & head, const int & n)
  {
    const_cast<Eigen::MatrixBase<T2> &>(output).template middleCols(head - n, n).noalias() = std::get<0>(m_weights) * input.template middleCols(head - ((KernelSize-1) * Dilation + n), n);
    for (int k = (KernelSize-2); k >= 0; --k)
    {
      const_cast<Eigen::MatrixBase<T2> &>(output).template middleCols(head - n, n) += m_weights[k] * input.template middleCols(head - ((k * Dilation)) + n, n);
    }
    
    const_cast<Eigen::MatrixBase<T2> &>(output).template middleCols(head - n, n).colwise() += m_bias;
  }

  template<typename T1, typename T2>
  inline void process2(Eigen::MatrixBase<T1> const & input, Eigen::MatrixBase<T2> const & output, const int & head, const int & n)
  {
    const_cast<Eigen::MatrixBase<T2> &>(output).template middleCols(head - n, n).noalias() =
        std::get<0>(m_weights) * input.template middleCols(head - ((KernelSize-1) * Dilation + n), n)
      + std::get<1>(m_weights) * input.template middleCols(head - ((KernelSize-2) * Dilation + n), n)
      + std::get<2>(m_weights) * input.template middleCols(head - ((KernelSize-3) * Dilation + n), n);

    
    const_cast<Eigen::MatrixBase<T2> &>(output).template middleCols(head - n, n).colwise() += m_bias;
  }

  template<typename T1, typename T2>
  inline void process3(Eigen::MatrixBase<T1> const & input, Eigen::MatrixBase<T2> const & output, const int & head, const int & n)
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
  conv1d_1<float, 3, 16, 1024> c;
  Eigen::Matrix<float, 16, 4096> input = Eigen::Matrix<float, 16, 4096>::Ones();
  Eigen::Matrix<float, 16, 4096> output;

  for (auto _ : state)
  {
    c.process1(input, output, 4096, 64);
    escape(&output);
  }
}

static void two(benchmark::State & state)
{
  conv1d_1<float, 3, 16, 1024> c;
  Eigen::Matrix<float, 16, 4096> input = Eigen::Matrix<float, 16, 4096>::Ones();
  Eigen::Matrix<float, 16, 4096> output;

  for (auto _ : state)
  {
    c.process2(input, output, 4096, 64);
    escape(&output);
  }
}

static void three(benchmark::State & state)
{
  conv1d_1<float, 3, 16, 1024> c;
  Eigen::Matrix<float, 16, 4096> input = Eigen::Matrix<float, 16, 4096>::Ones();
  Eigen::Matrix<float, 16, 4096> output;

  for (auto _ : state)
  {
    c.process3(input, output, 4096, 64);
    escape(&output);
  }
}

BENCHMARK(standard);
BENCHMARK(one);
BENCHMARK(two);
BENCHMARK(three);

BENCHMARK_MAIN();
