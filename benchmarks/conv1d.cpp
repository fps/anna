#include <Eigen/Core>
#include <benchmark/benchmark.h>
#include <anna/make_matrix_array.hpp>

#include <iostream>

template<int KernelSize, int Dilation, bool UseSpecialization>
struct conv1d
{
  template<typename T1, typename T2, typename T3>
  static void inline process(T1 const & weights, Eigen::MatrixBase<T2> const & input, Eigen::MatrixBase<T3> const & const_output, const int n, const int input_head, const int output_head)
  {
    Eigen::MatrixBase<T3> & output = const_cast<Eigen::MatrixBase<T3> &>(const_output);
    output.middleCols(output_head - n, n).noalias() = weights[0] * input.template middleCols(input_head - (KernelSize - 1  + n), n);
    for (int k = 1; k < KernelSize; ++k)
    {
      output.middleCols(output_head - n, n).noalias() += weights[k] * input.template middleCols(input_head - ((KernelSize - k - 1) * Dilation + n), n);
    }
  }

  template<typename T1, typename T2, typename T3, typename T4>
  static void inline process_with_bias(T1 const & weights, Eigen::MatrixBase<T2> const & bias, Eigen::MatrixBase<T3> const & input, Eigen::MatrixBase<T4> const & const_output, const int n, const int input_head, const int output_head)
  {
    Eigen::MatrixBase<T4> & output = const_cast<Eigen::MatrixBase<T4> &>(const_output);
    process(weights, input, output, n, input_head, output_head);
    output.template middleCols(output_head - n, n).template colwise() += bias;
  }

  template<typename T1, typename T2, typename T3, typename T4>
  static void inline process_with_bias2(T1 const & weights, Eigen::MatrixBase<T2> const & bias, Eigen::MatrixBase<T3> const & input, Eigen::MatrixBase<T4> const & const_output, const int n, const int input_head, const int output_head)
  {
    Eigen::MatrixBase<T3> & output = const_cast<Eigen::MatrixBase<T3> &>(const_output);
    output.middleCols(output_head - n, n).noalias() = weights[0] * input.template middleCols(input_head - ((KernelSize - 1) + n), n);
    for (int k = 1; k < KernelSize; ++k)
    {
      output.middleCols(output_head - n, n).noalias() += weights[k] * input.template middleCols(input_head - ((KernelSize - k - 1) * Dilation + n), n);
    }
    output.template middleCols(output_head - n, n).template colwise() += bias;
  }
};

template<int Dilation>
struct conv1d<3, Dilation, true>
{
  template<typename T1, typename T2, typename T3>
  static void inline process(T1 const & weights, Eigen::MatrixBase<T2> const & input, Eigen::MatrixBase<T3> const & const_output, const int n, const int input_head, const int output_head)
  {
    Eigen::MatrixBase<T3> & output = const_cast<Eigen::MatrixBase<T3> &>(const_output);
    output.middleCols(output_head - n, n).noalias() = 
        (std::get<0>(weights) * input.template middleCols(input_head - ((Dilation * 2) + n), n))
      + (std::get<1>(weights) * input.template middleCols(input_head - (Dilation + n), n))
      + (std::get<2>(weights) * input.template middleCols(input_head - n, n));
  }

  template<typename T1, typename T2, typename T3, typename T4>
  static void inline process_with_bias(T1 const & weights, Eigen::MatrixBase<T2> const & bias, Eigen::MatrixBase<T3> const & input, Eigen::MatrixBase<T4> const & const_output, const int n, const int input_head, const int output_head)
  {
    Eigen::MatrixBase<T4> & output = const_cast<Eigen::MatrixBase<T4> &>(const_output);
    process(weights, input, output, n, input_head, output_head);
    output.template middleCols(output_head - n, n).template colwise() += bias;
  }

  template<typename T1, typename T2, typename T3>
  static void inline process_d(T1 const & weights, const int dilation, Eigen::MatrixBase<T2> const & input, Eigen::MatrixBase<T3> const & const_output, const int n, const int input_head, const int output_head)
  {
    Eigen::MatrixBase<T3> & output = const_cast<Eigen::MatrixBase<T3> &>(const_output);
    output.middleCols(output_head - n, n).noalias() = 
        (std::get<0>(weights) * input.template middleCols(input_head - ((dilation * 2) + n), n))
      + (std::get<1>(weights) * input.template middleCols(input_head - (dilation + n), n))
      + (std::get<2>(weights) * input.template middleCols(input_head - n, n));
  }

  template<typename T1, typename T2, typename T3, typename T4>
  static void inline process_with_bias_d(T1 const & weights, Eigen::MatrixBase<T2> const & bias, const int dilation, Eigen::MatrixBase<T3> const & input, Eigen::MatrixBase<T4> const & const_output, const int n, const int input_head, const int output_head)
  {
    Eigen::MatrixBase<T4> & output = const_cast<Eigen::MatrixBase<T4> &>(const_output);
    process_d(weights, dilation, input, output, n, input_head, output_head);
    output.template middleCols(output_head - n, n).template colwise() += bias;
  }

  template<typename T1, typename T2, typename T3, typename T4>
  static void inline process_with_bias2(T1 const & weights, Eigen::MatrixBase<T2> const & bias, Eigen::MatrixBase<T3> const & input, Eigen::MatrixBase<T4> const & const_output, const int n, const int input_head, const int output_head)
  {
    Eigen::MatrixBase<T4> & output = const_cast<Eigen::MatrixBase<T4> &>(const_output);
    output.middleCols(output_head - n, n).noalias() = 
          (std::get<0>(weights) * input.template middleCols(input_head - ((Dilation * 2) + n), n))
        + (std::get<1>(weights) * input.template middleCols(input_head - (Dilation + n), n))
        + (std::get<2>(weights) * input.template middleCols(input_head - n, n));
    output.middleCols(output_head - n, n).colwise() += bias;
  }
};

template<int N, int Dilation, bool UseSpecialization>
static void run_with_bias_d(benchmark::State & state)
{
  const std::array<Eigen::Matrix<float, 8, 8>, 3> weights = anna::make_matrix_array<float, 8, 8, 3>(1.0);
  const Eigen::Matrix<float, 8, 4096> input = Eigen::Matrix<float, 8, 4096>::Ones();
  const Eigen::Vector<float, 8> bias = Eigen::Vector<float, 8>::Ones();
  Eigen::Matrix<float, 8, 4096> output = Eigen::Matrix<float, 8, 4096>::Zero();

  for (auto _ : state)
  {
    for (int n = 0; n < 750; ++n)
    {
      conv1d<3, Dilation, UseSpecialization>::template process_with_bias_d(weights, bias, Dilation, input, output, N, 4096, 4096);
    }
  }
}

template<int N, int Dilation, bool UseSpecialization>
static void run_with_bias(benchmark::State & state)
{
  const std::array<Eigen::Matrix<float, 8, 8>, 3> weights = anna::make_matrix_array<float, 8, 8, 3>(1.0);
  const Eigen::Matrix<float, 8, 4096> input = Eigen::Matrix<float, 8, 4096>::Ones();
  const Eigen::Vector<float, 8> bias = Eigen::Vector<float, 8>::Ones();
  Eigen::Matrix<float, 8, 4096> output = Eigen::Matrix<float, 8, 4096>::Zero();

  for (auto _ : state)
  {
    for (int n = 0; n < 750; ++n)
    {
      conv1d<3, Dilation, UseSpecialization>::template process_with_bias(weights, bias, input, output, N, 4096, 4096);
    }
  }
}

template<int N, int Dilation, bool UseSpecialization>
static void run_with_bias2(benchmark::State & state)
{
  const std::array<Eigen::Matrix<float, 8, 8>, 3> weights = anna::make_matrix_array<float, 8, 8, 3>(1.0);
  const Eigen::Matrix<float, 8, 4096> input = Eigen::Matrix<float, 8, 4096>::Ones();
  const Eigen::Vector<float, 8> bias = Eigen::Vector<float, 8>::Ones();
  Eigen::Matrix<float, 8, 4096> output = Eigen::Matrix<float, 8, 4096>::Zero();

  for (auto _ : state)
  {
    for (int n = 0; n < 750; ++n)
    {
      conv1d<3, Dilation, UseSpecialization>::template process_with_bias2(weights, bias, input, output, N, 4096, 4096);
    }
  }
}

template<int N, int Dilation, bool UseSpecialization>
static void run_d(benchmark::State & state)
{
  const std::array<Eigen::Matrix<float, 8, 8>, 3> weights = anna::make_matrix_array<float, 8, 8, 3>(1.0);
  const Eigen::Matrix<float, 8, 4096> input = Eigen::Matrix<float, 8, 4096>::Ones();
  Eigen::Matrix<float, 8, 4096> output = Eigen::Matrix<float, 8, 4096>::Zero();

  for (auto _ : state)
  {
    for (int n = 0; n < 750; ++n)
    {
      conv1d<3, Dilation, UseSpecialization>::template process_d(weights, Dilation, input, output, N, 4096, 4096);
    }
  }
}

template<int N, int Dilation, bool UseSpecialization>
static void run(benchmark::State & state)
{
  const std::array<Eigen::Matrix<float, 8, 8>, 3> weights = anna::make_matrix_array<float, 8, 8, 3>(1.0);
  const Eigen::Matrix<float, 8, 4096> input = Eigen::Matrix<float, 8, 4096>::Ones();
  Eigen::Matrix<float, 8, 4096> output = Eigen::Matrix<float, 8, 4096>::Zero();

  for (auto _ : state)
  {
    for (int n = 0; n < 750; ++n)
    {
      conv1d<3, Dilation, UseSpecialization>::template process(weights, input, output, N, 4096, 4096);
    }
  }
}


static const bool UseSpecialization = true;
static const bool UseNoSpecialization = false;

BENCHMARK(run<64, 1024, UseSpecialization>);
BENCHMARK(run_with_bias<64, 1024, UseSpecialization>);
BENCHMARK(run_with_bias2<64, 1024, UseSpecialization>);

BENCHMARK(run_d<64, 1024, UseSpecialization>);
BENCHMARK(run_with_bias_d<64, 1024, UseSpecialization>);

BENCHMARK(run<64, 1024, UseNoSpecialization>);
BENCHMARK(run_with_bias<64, 1024, UseNoSpecialization>);
BENCHMARK(run_with_bias2<64, 1024, UseNoSpecialization>);

BENCHMARK_MAIN();

