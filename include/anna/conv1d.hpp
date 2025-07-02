#pragma once

#include <Eigen/Core>
#include <array>
#include <cassert>
#include <vector>

namespace anna
{
  template<typename T, int k, int inp, int outp>
  std::array<Eigen::Matrix<T, outp, inp>, k>
  make_weights()
  {
    std::array<Eigen::Matrix<T, outp, inp>, k> w;
    w.fill(Eigen::Matrix<T, outp, inp>::Constant(1.0f / 24.0f));
    return w;
  }
  
  template<
    typename T,
    int N = 64,
    int kernel_size = 3,
    int in_channels = 1,
    int out_channels = 1,
    bool bias = false,
    int dilation = 1,
    int groups = 1
    >
  struct conv1d {
    typedef Eigen::Matrix<T, in_channels, 2 * ((kernel_size - 1) * dilation + N)> state_type;
    typedef Eigen::Vector<T, out_channels> bias_type;
    typedef std::array<Eigen::Matrix<T, out_channels, in_channels>, kernel_size> weights_type;
    
    conv1d() :
      m_weights(make_weights<T, kernel_size, in_channels, out_channels>()),
      m_bias(bias_type::Zero()),
      m_state(state_type::Zero()),
      m_state2(state_type::Zero()),
      m_state_size(m_state.ColsAtCompileTime),
      m_state_head(0),
      m_state_head2(m_state.ColsAtCompileTime / 2)
    {
      // std::cout << "State size: " << m_state.ColsAtCompileTime << ", m_state_head: " << m_state_head << ", m_state_head2: " << m_state_head2 << "\n";
    }

    weights_type m_weights;
    bias_type m_bias;

    state_type m_state;
    state_type m_state2;    

    const int m_state_size;

    int m_state_head;
    int m_state_head2;

    inline void advance(const int n)
    {
      m_state_head += n;
      m_state_head %= m_state_size;

      m_state_head2 += n;
      m_state_head2 %= m_state_size;
    }

    template<typename Matrix, typename Matrix2>
    inline void process(Eigen::MatrixBase<Matrix> const & input, Eigen::MatrixBase<Matrix2> const & output, const int n)
    // inline void process(const Eigen::Matrix<T, in_channels, N>  &input, const int n)
    {
      assert(n <= N);

      if (m_state_head + n < m_state_size) {
        m_state.middleCols(m_state_head, n).noalias() = input.leftCols(n);
      }
      else {
        const int m = m_state_size - m_state_head;
        m_state.middleCols(m_state_head, m).noalias() = input.leftCols(m);
        m_state.middleCols(0, m_state_head + n - m_state_size).noalias() = input.middleCols(m, n - m);
      }

      if (m_state_head2 + n < m_state_size) {
        m_state2.middleCols(m_state_head2, n).noalias() = input.leftCols(n);
      }
      else {
        const int m = m_state_size - m_state_head2;
        m_state2.middleCols(m_state_head2, m).noalias() = input.leftCols(m);
        m_state2.middleCols(0, m_state_head2 + n - m_state_size).noalias() = input.middleCols(m, n - m);
      }
      
      advance(n);

      if (m_state_head >= n) {
        const_cast<Eigen::MatrixBase<Matrix2>&>(output).template leftCols(n).noalias() = m_weights[0] * m_state.middleCols(m_state_head - n, n);
      }
      else {
        const_cast<Eigen::MatrixBase<Matrix2>&>(output).template leftCols(n).noalias() = m_weights[0] * m_state2.middleCols(m_state_head2 - n, n);
      }
      
      for (int k = 1; k < kernel_size; ++k) {
        if (m_state_head >= (n + k * dilation)) {
          const_cast<Eigen::MatrixBase<Matrix2>&>(output).template leftCols(n).noalias() += m_weights[k] * m_state.middleCols(m_state_head - (n + k * dilation), n);
        }
        else {
          const_cast<Eigen::MatrixBase<Matrix2>&>(output).template leftCols(n).noalias() += m_weights[k] * m_state2.middleCols(m_state_head2 - (n + k * dilation), n);            
        }
      }

      if constexpr(bias) {
        const_cast<Eigen::MatrixBase<Matrix2>&>(output).template leftCols(n).colwise() += m_bias;
      }
    }

    template <typename Matrix>
    inline void process_inplace(Eigen::MatrixBase<Matrix> const & input, const int n)
    {
      process(input, input, n);
    }

    template <typename Matrix>
    inline auto process(Eigen::MatrixBase<Matrix> const & input, const int n)
    {
      Eigen::Matrix<T, out_channels, N> ret;
      process(input, ret, n);
      return ret;
    }

    void set_parameters(std::vector<T> const & params, size_t & idx)
    {
      for (int row = 0; row < out_channels; ++row)
      {
        for (int col = 0; col < in_channels; ++col)
        {
          for (int k = 0; k < kernel_size; ++k)
          {
             m_weights[k](row, col) = params.at(idx++);
          }
        }
      }

      for (int row = 0; row < out_channels; ++row)
      {
        m_bias(row) = params.at(idx++);
      }
    }
  };
  
} // namespace nn
