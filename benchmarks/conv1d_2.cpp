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
  static void inline process(T1 const & weights, Eigen::MatrixBase<T2> const & input, Eigen::MatrixBase<T3> const & const_output, const int n, const int input_head, const int output_head)
  {
    Eigen::MatrixBase<T3> & output = const_cast<Eigen::MatrixBase<T3> &>(const_output);

    if constexpr (DoZero)
    {
      output.middleCols(output_head - n, n).noalias() = 
          (std::get<0>(weights) * input.template middleCols(input_head - n, n))
        + (std::get<1>(weights) * input.template middleCols(input_head - (Dilation + n), n))
        + (std::get<2>(weights) * input.template middleCols(input_head - ((Dilation << 1) + n), n));
    }
    else
    {
      output.middleCols(output_head - n, n).noalias() += 
          (std::get<0>(weights) * input.template middleCols(input_head - n, n))
        + (std::get<1>(weights) * input.template middleCols(input_head - (Dilation + n), n))
        + (std::get<2>(weights) * input.template middleCols(input_head - ((Dilation << 1) + n), n));
    }
  }

  template<typename T1, typename T2, typename T3, typename T4>
  static void inline process_with_bias(T1 const & weights, Eigen::MatrixBase<T2> const & bias, Eigen::MatrixBase<T3> const & input, Eigen::MatrixBase<T4> const & const_output, const int n, const int input_head, const int output_head)
  {
    Eigen::MatrixBase<T4> & output = const_cast<Eigen::MatrixBase<T4> &>(const_output);
    process(weights, input, output, n, input_head, output_head);
    output.template middleCols(output_head - n, n).template colwise() += bias;
  }
};

template<int N, int Dilation, bool DoZero, bool DoBias>
static void run(benchmark::State & state)
{
  const std::array<Eigen::Matrix<float, 16, 16>, 3> weights = anna::make_weights<float, 3, 16, 16>(1.0);
  const Eigen::Matrix<float, 16, 4096> input = Eigen::Matrix<float, 16, 4096>::Ones();
  const Eigen::Vector<float, 16> bias = Eigen::Vector<float, 16>::Ones();
  Eigen::Matrix<float, 16, 4096> output = Eigen::Matrix<float, 16, 4096>::Zero();

  for (auto _ : state)
  {
    if constexpr (DoBias)
    {
      conv1d<3, Dilation, DoZero>::template process_with_bias(weights, bias, input, output, N, 4096, 4096);
    }
    else
    {
      conv1d<3, Dilation, DoZero>::template process(weights, input, output, N, 4096, 4096);
    }
    escape(&output);
  }
}

static const bool DoZero = true;
static const bool DoNotZero = false;
static const bool DoBias = true;
static const bool DoNoBias = false;

BENCHMARK(run<64, 1024, DoNotZero, DoNoBias>);
BENCHMARK(run<64, 1024, DoNotZero, DoBias>);
BENCHMARK(run<64, 1024, DoZero, DoNoBias>);
BENCHMARK(run<64, 1024, DoZero, DoBias>);

BENCHMARK_MAIN();

