#include <Eigen/Core>
#include <benchmark/benchmark.h>
#include <anna/make_matrix_array.hpp>

#include <iostream>

template<int KernelSize, int Dilation, int N>
struct conv1d_fixed
{
  template<typename T1, typename T2, typename T3>
  static void inline process(T1 const & weights, Eigen::MatrixBase<T2> const & input, Eigen::MatrixBase<T3> const & const_output, const int input_head, const int output_head)
  {
    Eigen::MatrixBase<T3> & output = const_cast<Eigen::MatrixBase<T3> &>(const_output);
    output.template middleCols<N>(output_head - N).noalias() = weights[0] * input.template middleCols<N>(input_head - ((KernelSize - 1) * Dilation + N));
    for (int k = 1; k < KernelSize; ++k)
    {
      output.template middleCols<N>(output_head - N).noalias() += weights[k] * input.template middleCols<N>(input_head - ((KernelSize - k - 1) * Dilation + N));
    }
  }

  template<typename T1, typename T2, typename T3, typename T4>
  static void inline process_with_bias(T1 const & weights, Eigen::MatrixBase<T2> const & bias, Eigen::MatrixBase<T3> const & input, Eigen::MatrixBase<T4> const & const_output, const int input_head, const int output_head)
  {
    Eigen::MatrixBase<T4> & output = const_cast<Eigen::MatrixBase<T4> &>(const_output);
    process(weights, input, output, input_head, output_head);
    output.template middleCols<N>(output_head - N).template colwise() += bias;
  }

  template<typename T1, typename T2, typename T3, typename T4>
  static void inline process_with_bias2(T1 const & weights, Eigen::MatrixBase<T2> const & bias, Eigen::MatrixBase<T3> const & input, Eigen::MatrixBase<T4> const & const_output, const int input_head, const int output_head)
  {
    Eigen::MatrixBase<T3> & output = const_cast<Eigen::MatrixBase<T3> &>(const_output);
    output.template middleCols<N>(output_head - N).noalias() = weights[0] * input.template middleCols<N>(input_head - ((KernelSize - 1) * Dilation + N));
    for (int k = 1; k < KernelSize; ++k)
    {
      output.template middleCols<N>(output_head - N).noalias() += weights[k] * input.template middleCols<N>(input_head - ((KernelSize - k - 1) * Dilation + N));
    }
    output.template middleCols<N>(output_head - N).template colwise() += bias;
  }
};

template<int N, int Dilation>
static void run_fixed(benchmark::State & state)
{
  const std::array<Eigen::Matrix<float, 8, 8>, 3> weights = anna::make_matrix_array<float, 8, 8, 3>(1.0);
  const Eigen::Matrix<float, 8, 4096> input = Eigen::Matrix<float, 8, 4096>::Ones();
  Eigen::Matrix<float, 8, 4096> output = Eigen::Matrix<float, 8, 4096>::Zero();

  for (auto _ : state)
  {
    for (int n = 0; n < 750; ++n)
    {
      conv1d_fixed<3, Dilation, N>::template process(weights, input, output, 4096, 4096);
    }
  }
}

template<int N, int Dilation>
static void run_fixed_with_bias(benchmark::State & state)
{
  const std::array<Eigen::Matrix<float, 8, 8>, 3> weights = anna::make_matrix_array<float, 8, 8, 3>(1.0);
  const Eigen::Matrix<float, 8, 4096> input = Eigen::Matrix<float, 8, 4096>::Ones();
  const Eigen::Vector<float, 8> bias = Eigen::Vector<float, 8>::Ones();
  Eigen::Matrix<float, 8, 4096> output = Eigen::Matrix<float, 8, 4096>::Zero();

  for (auto _ : state)
  {
    for (int n = 0; n < 750; ++n)
    {
      conv1d_fixed<3, Dilation, N>::template process_with_bias(weights, bias, input, output, 4096, 4096);
    }
  }
}

template<int N, int Dilation>
static void run_fixed_with_bias2(benchmark::State & state)
{
  const std::array<Eigen::Matrix<float, 8, 8>, 3> weights = anna::make_matrix_array<float, 8, 8, 3>(1.0);
  const Eigen::Matrix<float, 8, 4096> input = Eigen::Matrix<float, 8, 4096>::Ones();
  const Eigen::Vector<float, 8> bias = Eigen::Vector<float, 8>::Ones();
  Eigen::Matrix<float, 8, 4096> output = Eigen::Matrix<float, 8, 4096>::Zero();

  for (auto _ : state)
  {
    for (int n = 0; n < 750; ++n)
    {
      conv1d_fixed<3, Dilation, N>::template process_with_bias2(weights, bias, input, output, 4096, 4096);
    }
  }
}

BENCHMARK(run_fixed<64, 1024>);
BENCHMARK(run_fixed_with_bias<64, 1024>);
BENCHMARK(run_fixed_with_bias2<64, 1024>);

BENCHMARK_MAIN();

