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
  
  template<typename T1, typename T2, typename T3>
  inline void process_conv1d(Eigen::MatrixBase<T1> const & weights, Eigen::MatrixBase<T2> const & input, Eigen::MatrixBase<T3> const & const_output)
  {
    Eigen::MatrixBase<T3> & output = const_cast<Eigen::MatrixBase<T3>&>(const_output);

    constexpr int KernelSize = weights.ColsAtCompileTime / input.ColsAtCompileTime;
  }

  template<typename T, int N = 64, int KernelSize = 3, int InChannels = 1, int OutChannels = 1, bool Bias = false, int Dilation = 1>
  struct conv1d {
    typedef Eigen::Vector<T, OutChannels> bias_type;
    typedef std::array<Eigen::Matrix<T, OutChannels, InChannels>, KernelSize> weights_type;
    
    weights_type m_weights;
    bias_type m_bias;

     conv1d() :
      m_weights(make_weights<T, KernelSize, InChannels, OutChannels>(0)),
      m_bias(bias_type::Zero())
    {

    }

   template<typename M1, typename M2>
    inline void process(Eigen::MatrixBase<M1> const & input, Eigen::MatrixBase<M2> const & output, const int input_head, const int output_head, const int n)
    {
      assert(n <= N);

      Eigen::MatrixBase<M2> & out = const_cast<Eigen::MatrixBase<M2>&>(output);

      out.middleCols(output_head - n, n).noalias() = m_weights[0] * input.middleCols(input_head - (n + (KernelSize-1) * Dilation), n);
      for (int k = 1; k < KernelSize; ++k) {
        out.middleCols(output_head - n, n).noalias() += m_weights[k] * input.middleCols(input_head - (n + (KernelSize-1-k) * Dilation), n);
      }

      if constexpr(Bias) {
        out.middleCols(output_head - n, n).colwise() += m_bias;
      }
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
  
} // namespace anna
