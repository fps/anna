#pragma once

#include <Eigen/Core>
#include <array>
#include <cassert>
#include <vector>

#include <anna/conv1d_make_weights.hpp>

namespace anna
{
  template<
    typename T,
    int N = 64,
    int KernelSize = 3,
    int InChannels = 1,
    int OutChannels = 1,
    bool Bias = false,
    int Dilation = 1
    >
  struct conv1d_double_buffer {
    /*
       We make the state twice as big as necessary to hold all samples required. And also
       we create the state _twice_ with one write head starting in the middle. This
       has the benefit of having _all_ required history available in a continuous
       fashion in one of the two states _always_.

       From this two nice properties result:
       1. We can do the matrix multiplication in one go.
       2. The timing should be pretty much the same everytime
    */
    typedef Eigen::Matrix<T, InChannels, 2 * ((KernelSize - 1) * Dilation + N)> state_type;
    typedef Eigen::Vector<T, OutChannels> bias_type;
    typedef std::array<Eigen::Matrix<T, OutChannels, InChannels>, KernelSize> weights_type;
    
    conv1d_double_buffer() :
      m_weights(make_weights<T, KernelSize, InChannels, OutChannels>(0)),
      m_bias(bias_type::Zero()),
      m_state(state_type::Zero()),
      m_state2(state_type::Zero()),
      m_state_size(m_state.ColsAtCompileTime),
      m_state_head(0),
      // m_state_head2 one starts in the middle. See above comment.
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

		Eigen::Matrix<T, OutChannels, N> m_output;

    inline void advance(const int n)
    {
      m_state_head += n;
      m_state_head %= m_state_size;

      m_state_head2 += n;
      m_state_head2 %= m_state_size;
    }

    template<typename M1, typename M2>
    inline void process(Eigen::MatrixBase<M1> const & input, Eigen::MatrixBase<M2> const & output, const int n)
    // inline void process(const Eigen::Matrix<T, InChannels, N>  &input, const int n)
    {
      assert(n <= N);

      Eigen::MatrixBase<M2> & out = const_cast<Eigen::MatrixBase<M2>&>(output);

      /*
        We fill both states with new incoming samples. 
      */
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

      /*
        Here we check which of the two states has _all_ required samples available
      */
      if (m_state_head >= m_state_size / 2) {
        out.leftCols(n).noalias() = m_weights[0] * m_state.middleCols(m_state_head - (n + (KernelSize-1) * Dilation), n);
        for (int k = 1; k < KernelSize; ++k) {
          out.leftCols(n).noalias() += m_weights[k] * m_state.middleCols(m_state_head - (n + (KernelSize-1-k) * Dilation), n);
        }
      }
      else {
        out.leftCols(n).noalias() = m_weights[0] * m_state2.middleCols(m_state_head2 - (n + (KernelSize-1) * Dilation), n);
        for (int k = 1; k < KernelSize; ++k) {
          out.leftCols(n).noalias() += m_weights[k] * m_state2.middleCols(m_state_head2 - (n + (KernelSize-1-k) * Dilation), n);
        }
      }

      if constexpr(Bias) {
        out.leftCols(n).colwise() += m_bias;
      }
    }

    template <typename M1>
    inline void process_inplace(Eigen::MatrixBase<M1> const & input, const int n)
    {
      process(input, input, n);
    }

    template <typename M1>
    inline void process_cached(Eigen::MatrixBase<M1> const & input, const int n)
    {
      process(input, m_output, n);
    }

    template <typename M1>
    inline auto process(Eigen::MatrixBase<M1> const & input, const int n)
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
