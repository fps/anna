#pragma once

#include <Eigen/Core>
#include <array>
#include <cassert>
#include <vector>

#include <anna/conv1d_make_weights.hpp>

namespace anna
{
  template<typename T, int N, int KernelSize, int InChannels, int OutChannels, bool Bias, int Dilation>
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
