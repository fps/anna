#include <benchmark/benchmark.h>


#include <anna/make_matrix_array.hpp>

template<typename WeightsType, typename InputType, typename OutputType>
static void inline conv1d(WeightsType const & weights, const int dilation, Eigen::MatrixBase<InputType> const & input, Eigen::MatrixBase<OutputType> const & const_output, const int n, const int input_head, const int output_head)
{
  constexpr int KernelSize = weights.size();
  Eigen::MatrixBase<OutputType> & output = const_cast<Eigen::MatrixBase<OutputType> &>(const_output);

  output.middleCols(output_head - n, n).noalias() = weights[0] * input.template middleCols(input_head - (KernelSize - 1  + n), n);
  for (int k = 1; k < KernelSize; ++k)
  {
    output.middleCols(output_head - n, n).noalias() += weights[k] * input.template middleCols(input_head - ((KernelSize - k - 1) * dilation + n), n);
  }
}

template<typename WeightsType, typename BiasType, typename InputType, typename OutputType>
static void inline conv1d(WeightsType const & weights, Eigen::MatrixBase<BiasType> const & bias, const int dilation, Eigen::MatrixBase<InputType> const & input, Eigen::MatrixBase<OutputType> const & const_output, const int n, const int input_head, const int output_head)
{
  Eigen::MatrixBase<OutputType> & output = const_cast<Eigen::MatrixBase<OutputType> &>(const_output);
  
  conv1d(weights, dilation, input, output, n, input_head, output_head);
  output.middleCols(output_head - n, n).colwise() += bias;
}

template<int N, int Dilation, bool DoBias>
static void run(benchmark::State & state)
{
  const std::array<Eigen::Matrix<float, 8, 8>, 3> weights = anna::make_matrix_array<float, 8, 8, 3>(1.0);
  const Eigen::Vector<float, 8> bias = Eigen::Vector<float, 8>::Ones();
  const Eigen::Matrix<float, 8, 4096> input = Eigen::Matrix<float, 8, 4096>::Ones();
  Eigen::Matrix<float, 8, 4096> output = Eigen::Matrix<float, 8, 4096>::Zero();

  for (auto _ : state)
  {
    for (int n = 0; n < 750; ++n)
    {
      if constexpr(DoBias) 
      {
        conv1d(weights, bias, Dilation, input, output, N, 4096, 4096);
      }
      else
      {
        conv1d(weights, Dilation, input, output, N, 4096, 4096);
      }
    }
  }
}

BENCHMARK(run<64, 1024, false>);
BENCHMARK(run<64, 1024, true>);
BENCHMARK_MAIN();
