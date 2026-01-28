#include <Eigen/Core>
#include <benchmark/benchmark.h>
#include <anna/benchmark.hpp>
#include <anna/conv1d_make_weights.hpp>

#include <iostream>

template<int KernelSize, int Dilation, int N>
struct conv1d_fixed
{
  template<typename T1, typename T2, typename T3>
  static void inline process(T1 const & weights, Eigen::MatrixBase<T2> const & input, Eigen::MatrixBase<T3> const & const_output, const int input_head, const int output_head)
  {
    Eigen::MatrixBase<T3> & output = const_cast<Eigen::MatrixBase<T3> &>(const_output);
    output.template middleCols<N>(output_head - N).noalias() = weights[0] * input.template middleCols<N>(input_head - N);
    for (int k = 1; k < KernelSize; ++k)
    {
      output.template middleCols<N>(output_head - N).noalias() += weights[k] * input.template middleCols<N>(input_head - (k * Dilation + N));
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
    output.template middleCols<N>(output_head - N).noalias() = weights[0] * input.template middleCols<N>(input_head - N);
    for (int k = 1; k < KernelSize; ++k)
    {
      output.template middleCols<N>(output_head - N).noalias() += weights[k] * input.template middleCols<N>(input_head - (k * Dilation + N));
    }
    output.template middleCols<N>(output_head - N).template colwise() += bias;
  }
};

template<int KernelSize, int Dilation, bool UseSpecialization>
struct conv1d
{
  template<typename T1, typename T2, typename T3>
  static void inline process(T1 const & weights, Eigen::MatrixBase<T2> const & input, Eigen::MatrixBase<T3> const & const_output, const int n, const int input_head, const int output_head)
  {
    Eigen::MatrixBase<T3> & output = const_cast<Eigen::MatrixBase<T3> &>(const_output);
    output.middleCols(output_head - n, n).noalias() = weights[0] * input.template middleCols(input_head - n, n);
    for (int k = 1; k < KernelSize; ++k)
    {
      output.middleCols(output_head - n, n).noalias() += weights[k] * input.template middleCols(input_head - (k * Dilation + n), n);
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
    output.middleCols(output_head - n, n).noalias() = weights[0] * input.template middleCols(input_head - n, n);
    for (int k = 1; k < KernelSize; ++k)
    {
      output.middleCols(output_head - n, n).noalias() += weights[k] * input.template middleCols(input_head - (k * Dilation + n), n);
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
        (std::get<0>(weights) * input.template middleCols(input_head - n, n))
      + (std::get<1>(weights) * input.template middleCols(input_head - (Dilation + n), n))
      + (std::get<2>(weights) * input.template middleCols(input_head - ((Dilation * 2) + n), n));
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
    Eigen::MatrixBase<T4> & output = const_cast<Eigen::MatrixBase<T4> &>(const_output);
    output.middleCols(output_head - n, n).noalias() = 
          (std::get<0>(weights) * input.template middleCols(input_head - n, n))
        + (std::get<1>(weights) * input.template middleCols(input_head - (Dilation + n), n))
        + (std::get<2>(weights) * input.template middleCols(input_head - ((Dilation * 2) + n), n));
    output.middleCols(output_head - n, n).colwise() += bias;
  }
};

template<int N, int Dilation, bool UseSpecialization>
static void run_with_bias(benchmark::State & state)
{
  const std::array<Eigen::Matrix<float, 16, 16>, 3> weights = anna::make_weights<float, 3, 16, 16>(1.0);
  const Eigen::Matrix<float, 16, 4096> input = Eigen::Matrix<float, 16, 4096>::Ones();
  const Eigen::Vector<float, 16> bias = Eigen::Vector<float, 16>::Ones();
  Eigen::Matrix<float, 16, 4096> output = Eigen::Matrix<float, 16, 4096>::Zero();

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
  const std::array<Eigen::Matrix<float, 16, 16>, 3> weights = anna::make_weights<float, 3, 16, 16>(1.0);
  const Eigen::Matrix<float, 16, 4096> input = Eigen::Matrix<float, 16, 4096>::Ones();
  const Eigen::Vector<float, 16> bias = Eigen::Vector<float, 16>::Ones();
  Eigen::Matrix<float, 16, 4096> output = Eigen::Matrix<float, 16, 4096>::Zero();

  for (auto _ : state)
  {
    for (int n = 0; n < 750; ++n)
    {
      conv1d<3, Dilation, UseSpecialization>::template process_with_bias2(weights, bias, input, output, N, 4096, 4096);
    }
  }
}

template<int N, int Dilation, bool UseSpecialization>
static void run(benchmark::State & state)
{
  const std::array<Eigen::Matrix<float, 16, 16>, 3> weights = anna::make_weights<float, 3, 16, 16>(1.0);
  const Eigen::Matrix<float, 16, 4096> input = Eigen::Matrix<float, 16, 4096>::Ones();
  Eigen::Matrix<float, 16, 4096> output = Eigen::Matrix<float, 16, 4096>::Zero();

  for (auto _ : state)
  {
    for (int n = 0; n < 750; ++n)
    {
      conv1d<3, Dilation, UseSpecialization>::template process(weights, input, output, N, 4096, 4096);
    }
  }
}

template<int N, int Dilation>
static void run_fixed(benchmark::State & state)
{
  const std::array<Eigen::Matrix<float, 16, 16>, 3> weights = anna::make_weights<float, 3, 16, 16>(1.0);
  const Eigen::Matrix<float, 16, 4096> input = Eigen::Matrix<float, 16, 4096>::Ones();
  Eigen::Matrix<float, 16, 4096> output = Eigen::Matrix<float, 16, 4096>::Zero();

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
  const std::array<Eigen::Matrix<float, 16, 16>, 3> weights = anna::make_weights<float, 3, 16, 16>(1.0);
  const Eigen::Matrix<float, 16, 4096> input = Eigen::Matrix<float, 16, 4096>::Ones();
  const Eigen::Vector<float, 16> bias = Eigen::Vector<float, 16>::Ones();
  Eigen::Matrix<float, 16, 4096> output = Eigen::Matrix<float, 16, 4096>::Zero();

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
  const std::array<Eigen::Matrix<float, 16, 16>, 3> weights = anna::make_weights<float, 3, 16, 16>(1.0);
  const Eigen::Matrix<float, 16, 4096> input = Eigen::Matrix<float, 16, 4096>::Ones();
  const Eigen::Vector<float, 16> bias = Eigen::Vector<float, 16>::Ones();
  Eigen::Matrix<float, 16, 4096> output = Eigen::Matrix<float, 16, 4096>::Zero();

  for (auto _ : state)
  {
    for (int n = 0; n < 750; ++n)
    {
      conv1d_fixed<3, Dilation, N>::template process_with_bias2(weights, bias, input, output, 4096, 4096);
    }
  }
}

static const bool UseSpecialization = true;
static const bool UseNoSpecialization = false;

BENCHMARK(run_fixed<64, 1024>);
BENCHMARK(run_fixed_with_bias<64, 1024>);
BENCHMARK(run_fixed_with_bias2<64, 1024>);

BENCHMARK(run<64, 1024, UseSpecialization>);
BENCHMARK(run_with_bias<64, 1024, UseSpecialization>);
BENCHMARK(run_with_bias2<64, 1024, UseSpecialization>);

BENCHMARK(run<64, 1024, UseNoSpecialization>);
BENCHMARK(run_with_bias<64, 1024, UseNoSpecialization>);
BENCHMARK(run_with_bias2<64, 1024, UseNoSpecialization>);

/*
static const bool DoZero = true;
static const bool DoNotZero = false;
static const bool DoBias = true;
static const bool DoNoBias = false;

BENCHMARK(run<64, 1024, DoNotZero, DoNoBias>);
BENCHMARK(run<64, 1024, DoNotZero, DoBias>);
*/
// BENCHMARK(run<64, 1024, DoZero, DoNoBias>);
// BENCHMARK(run<64, 1024, DoZero, DoBias>);

BENCHMARK_MAIN();

