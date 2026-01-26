#include <Eigen/Core>
#include <benchmark/benchmark.h>
#include <anna/conv1d_double_buffer.hpp>
#include <anna/benchmark.hpp>

template<int KernelSize, int Dilation = 1, bool DoZero = false>
struct conv1d
{
  template<typename T1, typename T2, typename T3>
  static void inline process(T1 const & weights, Eigen::MatrixBase<T2> const & input, Eigen::MatrixBase<T3> const & output)
  {
  }
};

template<int Dilation, bool DoZero>
struct conv1d<3, Dilation, DoZero>
{
  template<typename T1, typename T2, typename T3>
  static void inline process(T1 const & weights, Eigen::MatrixBase<T2> const & input, Eigen::MatrixBase<T3> const & const_output)
  {
    Eigen::MatrixBase<T3> & output = const_cast<Eigen::MatrixBase<T3> &>(const_output);

    if constexpr (DoZero)
    {
      output.noalias() = 
          (std::get<0>(weights) * input)
        + (std::get<1>(weights) * input)
        + (std::get<2>(weights) * input);
    }
    else
    {
      output.noalias() += 
          (std::get<0>(weights) * input)
        + (std::get<1>(weights) * input)
        + (std::get<2>(weights) * input);
    }
  }

  template<typename T1, typename T2, typename T3, typename T4>
  static void inline process_with_bias(T1 const & weights, Eigen::MatrixBase<T2> const & bias, Eigen::MatrixBase<T3> const & input, Eigen::MatrixBase<T4> const & const_output)
  {
    process(weights, input, const_output);
    Eigen::MatrixBase<T3> & output = const_cast<Eigen::MatrixBase<T3> &>(const_output);
    output.template colwise() += bias;
  }
};

template<int N, int Dilation, bool DoAdd, bool DoBias>
static void run(benchmark::State & state)
{
  const std::array<Eigen::Matrix<float, 16, 16>, 3> weights = anna::make_weights<float, 3, 16, 16>(1.0);
  const Eigen::Matrix<float, 16, N> input = Eigen::Matrix<float, 16, N>::Ones();
  const Eigen::Vector<float, 16> bias = Eigen::Vector<float, 16>::Ones();
  Eigen::Matrix<float, 16, N> output;

  for (auto _ : state)
  {
    if constexpr (DoBias)
    {
      conv1d<3, 1024, DoAdd>::template process_with_bias(weights, bias, input, output);
    }
    else
    {
      conv1d<3, 1024, DoAdd>::template process(weights, input, output);
    }
    escape(&output);
  }
}

BENCHMARK(run<64, 1, false, false>);

BENCHMARK(run<64, 1, false, true>);

BENCHMARK(run<64, 1, true, false>);

BENCHMARK(run<64, 1, true, true>);

BENCHMARK_MAIN();

