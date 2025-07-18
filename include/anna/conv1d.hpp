#pragma once

#include <Eigen/Core>
#include <array>
#include <cassert>
#include <vector>

namespace anna
{
  template<typename T, int k, int inp, int outp>
  std::array<Eigen::Matrix<T, outp, inp>, k>
  make_weights(T value)
  {
    std::array<Eigen::Matrix<T, outp, inp>, k> w;
    w.fill(Eigen::Matrix<T, outp, inp>::Constant(value));
    return w;
  }
  
  template<
    typename T,
    int N = 64,
    int KernelSize = 3,
    int InChannels = 1,
    int OutChannels = 1,
    bool Bias = false,
    int Dilation = 1
    >
  struct conv1d {
    typedef Eigen::Matrix<T, InChannels, 2 * ((KernelSize - 1) * Dilation + N)> state_type;
    typedef Eigen::Vector<T, OutChannels> bias_type;
    typedef std::array<Eigen::Matrix<T, OutChannels, InChannels>, KernelSize> weights_type;
    
    conv1d() :
      m_weights(make_weights<T, KernelSize, InChannels, OutChannels>(0)),
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
    // inline void process(const Eigen::Matrix<T, InChannels, N>  &input, const int n)
    {
      assert(n <= N);

      if (m_state_head + n <= m_state_size) {
        m_state.middleCols(m_state_head, n).noalias() = input.leftCols(n);
      }
      else {
        const int m = m_state_size - m_state_head;
        m_state.middleCols(m_state_head, m).noalias() = input.leftCols(m);
        m_state.middleCols(0, n - m).noalias() = input.middleCols(m, n - m);
      }

      if (m_state_head2 + n <= m_state_size) {
        m_state2.middleCols(m_state_head2, n).noalias() = input.leftCols(n);
      }
      else {
        const int m = m_state_size - m_state_head2;
        m_state2.middleCols(m_state_head2, m).noalias() = input.leftCols(m);
        m_state2.middleCols(0, n - m).noalias() = input.middleCols(m, n - m);
      }
      
      advance(n);

      if (m_state_head >= m_state_size / 2) {
        const_cast<Eigen::MatrixBase<Matrix2>&>(output).template leftCols(n).noalias() = m_weights[0] * m_state.middleCols(m_state_head - (n + (KernelSize-1) * Dilation), n);
        for (int k = 1; k < KernelSize; ++k) {
          const_cast<Eigen::MatrixBase<Matrix2>&>(output).template leftCols(n).noalias() += m_weights[k] * m_state.middleCols(m_state_head - (n + (KernelSize-1-k) * Dilation), n);
        }
      }
      else {
        const_cast<Eigen::MatrixBase<Matrix2>&>(output).template leftCols(n).noalias() = m_weights[0] * m_state2.middleCols(m_state_head2 - (n + (KernelSize-1) * Dilation), n);
        for (int k = 1; k < KernelSize; ++k) {
          const_cast<Eigen::MatrixBase<Matrix2>&>(output).template leftCols(n).noalias() += m_weights[k] * m_state2.middleCols(m_state_head2 - (n + (KernelSize-1-k) * Dilation), n);
        }
      }

      if constexpr(Bias) {
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
      Eigen::Matrix<T, OutChannels, N> ret;
      process(input, ret, n);
      return ret;
    }

    void set_parameters(std::vector<T> const & params, size_t & idx)
    {
      for (int row = 0; row < OutChannels; ++row)
      {
        for (int col = 0; col < InChannels; ++col)
        {
          for (int k = 0; k < KernelSize; ++k)
          {
             m_weights[k](row, col) = params.at(idx++);
          }
        }
      }

      if constexpr(Bias)
      {
        for (int row = 0; row < OutChannels; ++row)
        {
          m_bias(row) = params.at(idx++);
        }
      }
    }
  };
  
} // namespace nn
