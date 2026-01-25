#include <Eigen/Core>
#include <benchmark/benchmark.h>
// #include <anna/conv1d.hpp>
#include <anna/benchmark.hpp>

template<int KernelSize, int Dilation>
struct conv1d
{
  template<typename T1, typename T2>
  static inline auto process(Eigen::MatrixBase<T1> const & weights, Eigen::MatrixBase<T2> const & input)
  {
    /*
    if constexpr (KernelSize == 1)
    {
      return weights.template leftCols<input.RowsAtCompileTime>() * input;
    }
    else
    {
      return weights.template leftCols<input.RowsAtCompileTime>() * input + conv1d<KernelSize - 1, Dilation>::process(weights, input);
    }
    */
    if constexpr (KernelSize == 1)
    {
      return weights * input;
    }
    else
    {
      return weights * input + conv1d<KernelSize - 1, Dilation>::process(weights, input);
    }
  }
};

template<int Dilation>
struct conv1d<3, Dilation>
{
  template<typename T1, typename T2>
  static inline auto process(Eigen::MatrixBase<T1> const & weights, Eigen::MatrixBase<T2> const & input)
  {
    return 
        weights * input 
      + weights * input 
      + weights * input;
      /*
        weights.template leftCols<input.RowsAtCompileTime>() * input 
      + weights.template leftCols<input.RowsAtCompileTime>() * input 
      + weights.template leftCols<input.RowsAtCompileTime>() * input;
      */
  }
};



template<int N> 
static void run(benchmark::State & state)
{
  const Eigen::Matrix<float, 16, 16> weights = Eigen::Matrix<float, 16, 16>::Ones();
  const Eigen::Matrix<float, 16, N> input = Eigen::Matrix<float, 16, N>::Ones();
  Eigen::Matrix<float, 16, N> output;

  for (auto _ : state)
  {
    // process_conv1d<3, Eigen::Matrix<float, 16, 16*3>, Eigen::Matrix<float, 16, 4096>, Eigen::Matrix<float, 16, 4096>(weights, input, output);
    // process_conv1d(weights, input, integer<3>(), integer<1024>());
    output = conv1d<3, 1024>::template process(weights, input);
    escape(&output);
  }
}

BENCHMARK(run<64>);
BENCHMARK(run<128>);
BENCHMARK(run<256>);
BENCHMARK(run<512>);

BENCHMARK_MAIN();
