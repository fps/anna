#pragma once

namespace anna
{
  template<typename T>
  inline auto eigen_fast_tanh(const Eigen::ArrayBase<T> &x)
  {
    // auto x = m.array();
    auto x2 = x*x;
    auto ax = x.abs();
    auto num = (2.45550750702956f + 2.45550750702956f * ax + (0.893229853513558f + 0.821226666969744f * ax) * x2);
    auto denax = (x + 0.814642734961073f * x * ax).abs();
    auto denom = (2.44506634652299f + (2.44506634652299f + x2) * denax);
    return x * (num / denom);
  }

  template<typename Matrix>
  inline void inplace_eigen_fast_tanh(Eigen::MatrixBase<Matrix> const & m)
  {
    // auto m = const_cast<Eigen::MatrixBase<Matrix>&>(in);
    auto x = m.array();
    auto x2 = x*x;
    auto ax = x.abs();
    auto num = (2.45550750702956f + 2.45550750702956f * ax + (0.893229853513558f + 0.821226666969744f * ax) * x2);
    auto denax = (x + 0.814642734961073f * x * ax).abs();
    auto denom = (2.44506634652299f + (2.44506634652299f + x2) * denax);
    const_cast<Eigen::MatrixBase<Matrix>&>(m) = x * (num / denom);
  }

  template<typename Matrix>
  inline void inplace_fast_tanh(Eigen::MatrixBase<Matrix> const & m)
  {
    auto m2 = m.array();
  }


  template<typename T, int N, int channels>
  struct tanh
  {
    Eigen::Matrix<T, channels, N> m_output;

    template<typename Matrix>
    inline void process(const Eigen::MatrixBase<Matrix> &input, const int n)
    {
      m_output.template leftCols(n) = eigen_fast_tanh(input.template leftCols(n));
    }
  };

} // namespace nn
