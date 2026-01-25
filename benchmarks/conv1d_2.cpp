#include <Eigen/Core>
#include <benchmark/benchmark.h>
#include <anna/conv1d_double_buffer.hpp>
#include <anna/benchmark.hpp>

template<int KernelSize, int Dilation>
struct conv1d
{
  template<typename T1, typename T2, typename T3>
  static void inline process(T1 const & weights, Eigen::MatrixBase<T2> const & input, Eigen::MatrixBase<T3> const & output)
  {
  }
};

template<int Dilation>
struct conv1d<3, Dilation>
{
  template<typename T1, typename T2, typename T3>
  static void inline process(T1 const & weights, Eigen::MatrixBase<T2> const & input, Eigen::MatrixBase<T3> const & const_output)
  {
    Eigen::MatrixBase<T3> & output = const_cast<Eigen::MatrixBase<T3> &>(const_output);
    output += 
        (std::get<0>(weights) * input)
      + (std::get<1>(weights) * input)
      + (std::get<2>(weights) * input);
  }
};

template<int N>
static void run(benchmark::State & state)
{
  const std::array<Eigen::Matrix<float, 16, 16>, 3> weights = anna::make_weights<float, 3, 16, 16>(1.0);
  const Eigen::Matrix<float, 16, N> input = Eigen::Matrix<float, 16, N>::Ones();
  Eigen::Matrix<float, 16, N> output;

  for (auto _ : state)
  {
    // process_conv1d<3, Eigen::Matrix<float, 16, 16*3>, Eigen::Matrix<float, 16, 4096>, Eigen::Matrix<float, 16, 4096>(weights, input, output);
    // process_conv1d(weights, input, integer<3>(), integer<1024>());
    conv1d<3, 1024>::template process(weights, input, output);
    escape(&output);
  }
}

BENCHMARK(run<64>);
BENCHMARK(run<128>);
BENCHMARK(run<256>);
BENCHMARK(run<512>);

BENCHMARK_MAIN();

