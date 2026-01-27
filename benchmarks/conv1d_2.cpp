#include <Eigen/Core>
#include <benchmark/benchmark.h>
#include <anna/benchmark.hpp>
#include <anna/conv1d_make_weights.hpp>

template<bool Add>
struct assign
{
  template<typename T1, typename T2>
  static inline void process(Eigen::MatrixBase<T1> const & arg, Eigen::MatrixBase<T2> const & const_result)
  {
    Eigen::MatrixBase<T2> & result = const_cast<Eigen::MatrixBase<T2> &>(const_result);
    if constexpr (Add)
    {
      result.noalias() += arg;
    }
    else
    {
      result.noalias() = arg;
    }
  }
};

template<int KernelSize, bool UseSpecialization>
struct conv1d
{
  template<typename T1, typename T2, typename T3>
  static void inline process(T1 const & weights, Eigen::MatrixBase<T2> const & input, Eigen::MatrixBase<T3> const & output)
  {
  }
};

template<int Dilation, typename Assign>
struct conv1d<3, Dilation, Assign>
{
  template<typename T1, typename T2, typename T3>
  static void inline process(T1 const & weights, Eigen::MatrixBase<T2> const & input, Eigen::MatrixBase<T3> const & const_output, const int n, const int input_head, const int output_head)
  {
    Eigen::MatrixBase<T3> & output = const_cast<Eigen::MatrixBase<T3> &>(const_output);
    /* 

    // output.middleCols(output_head - n, n).array() = 0;
    output.middleCols(output_head - n, n).noalias() += 
        (std::get<0>(weights) * input.template middleCols(input_head - n, n))
      + (std::get<1>(weights) * input.template middleCols(input_head - (Dilation + n), n))
      + (std::get<2>(weights) * input.template middleCols(input_head - ((Dilation * 2) + n), n));
    */
    Assign::process(
        (std::get<0>(weights) * input.template middleCols(input_head - n, n))
      + (std::get<1>(weights) * input.template middleCols(input_head - (Dilation + n), n))
      + (std::get<2>(weights) * input.template middleCols(input_head - ((Dilation * 2) + n), n)),
      output.middleCols(output_head - n, n)
    );

  }

  template<typename T1, typename T2, typename T3, typename T4>
  static void inline process_with_bias(T1 const & weights, Eigen::MatrixBase<T2> const & bias, Eigen::MatrixBase<T3> const & input, Eigen::MatrixBase<T4> const & const_output, const int n, const int input_head, const int output_head)
  {
    Eigen::MatrixBase<T4> & output = const_cast<Eigen::MatrixBase<T4> &>(const_output);
    process(weights, input, output, n, input_head, output_head);
    output.template middleCols(output_head - n, n).template colwise() += bias;
  }
};

template<int N, int Dilation>
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
      conv1d<3, Dilation, assign<true>>::template process_with_bias(weights, bias, input, output, N, 4096, 4096);
    }
  }
}

template<int N, int Dilation>
static void run(benchmark::State & state)
{
  const std::array<Eigen::Matrix<float, 16, 16>, 3> weights = anna::make_weights<float, 3, 16, 16>(1.0);
  const Eigen::Matrix<float, 16, 4096> input = Eigen::Matrix<float, 16, 4096>::Ones();
  Eigen::Matrix<float, 16, 4096> output = Eigen::Matrix<float, 16, 4096>::Zero();

  for (auto _ : state)
  {
    for (int n = 0; n < 750; ++n)
    {
      conv1d<3, Dilation, assign<false>>::template process(weights, input, output, N, 4096, 4096);
    }
  }
}

BENCHMARK(run_with_bias<64, 128>);
BENCHMARK(run_with_bias<64, 64>);
BENCHMARK(run<64, 128>);
BENCHMARK(run<64, 512>);
BENCHMARK(run<64, 64>);
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

