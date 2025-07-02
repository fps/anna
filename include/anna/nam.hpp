#pragma once

#include <anna/conv1d.hpp>
#include <anna/tanh.hpp>

#include <cassert>

namespace anna
{
  namespace nam
  {
    template<typename T, int N, int kernel_size, int bottom_in_channels, int channels, int dilation, bool last = false>
    struct wavenet_layer
    {
      typedef Eigen::Matrix<T, channels, N> intermediate_type;
      conv1d<T, N, kernel_size, channels, channels, true, dilation> m_dilated;
      typedef Eigen::Matrix<T, channels, bottom_in_channels> input_mixer_weights_type;
      input_mixer_weights_type m_input_mixer_weights;
      
      typedef Eigen::Matrix<T, channels, channels> linear_weights_type;
      linear_weights_type m_linear_weights;
      typedef Eigen::Vector<T, channels> linear_bias_type;
      linear_bias_type m_linear_bias;
      wavenet_layer() :
        m_input_mixer_weights(input_mixer_weights_type::Constant(1.0f / 24.0f)),
        m_linear_weights(linear_weights_type::Constant(1.0f / 24.0f)),
        m_linear_bias(linear_bias_type::Zero())
      {

      }
      
      template<
        typename Matrix1,
        typename Matrix2,
        typename Matrix3,
        typename Matrix4
        >
      inline void process(
        Eigen::MatrixBase<Matrix1> const & input,
        Eigen::MatrixBase<Matrix2> const & bottom_input,
        Eigen::MatrixBase<Matrix3> const & head,
        Eigen::MatrixBase<Matrix4> const & output, const int n)
      {
        m_dilated.process(input, output, n);
        const_cast<Eigen::MatrixBase<Matrix4>&>(output).template leftCols(n).noalias() += m_input_mixer_weights * bottom_input.leftCols(n);
        
        inplace_eigen_fast_tanh(const_cast<Eigen::MatrixBase<Matrix4>&>(output).template leftCols(n));
        
        const_cast<Eigen::MatrixBase<Matrix3>&>(head).template leftCols(n).noalias() += output.template leftCols(n);

        if constexpr(!last) {
          const_cast<Eigen::MatrixBase<Matrix4>&>(output).template leftCols(n) = (m_linear_weights * output.template leftCols(n)).colwise() + m_linear_bias;
        
          const_cast<Eigen::MatrixBase<Matrix4>&>(output).template leftCols(n).noalias() += input.template leftCols(n);
        }
      }
    };
    
    template<typename Layers, int remaining>
    struct process_wavenet_block
    {
      template<
        typename Matrix1,
        typename Matrix2,
        typename Matrix3,
        typename Matrix4
      >
      static void go(Layers &layers,
        Eigen::MatrixBase<Matrix1> const & input,
        Eigen::MatrixBase<Matrix2> const & bottom_input,
        Eigen::MatrixBase<Matrix3> const & head,
        Eigen::MatrixBase<Matrix4> const & output,
        const int n)
      {
        std::get<std::tuple_size_v<Layers> - remaining>(layers).process(input, bottom_input, head, output, n);
        process_wavenet_block<Layers, remaining-1>::go(layers, output, bottom_input, head, input, n);
        // const_cast<Eigen::MatrixBase<Matrix1>&>(input).template leftCols(n).noalias() = output.template leftCols(n);
      }
    };
    
    template<typename Layers>
    struct process_wavenet_block<Layers, 0>
    {
      template<
        typename Matrix1,
        typename Matrix2,
        typename Matrix3,
        typename Matrix4
      >
      static void go(Layers &layers,
        Eigen::MatrixBase<Matrix1> const & input,
        Eigen::MatrixBase<Matrix2> const & bottom_input,
        Eigen::MatrixBase<Matrix3> const & head,
        Eigen::MatrixBase<Matrix4> const & output,
        const int n)
      {
      }
    };
    template<typename T, int N, int bottom_in_channels, int in_channels, int channels, int head_output_channels, typename... Layers>
    struct wavenet_block
    {
      typedef std::tuple<Layers...> layers_type;
      layers_type m_layers;
      typedef Eigen::Matrix<T, channels, N> head_type;
      head_type m_head;
      typedef Eigen::Matrix<T, channels, N> output_type;
      output_type m_output;
      typedef Eigen::Matrix<T, channels, N> buffer_type;
      output_type m_buffer;
      typedef Eigen::Matrix<T, head_output_channels, N> head_output_type;
      head_output_type m_head_output;
      Eigen::Matrix<T, channels, in_channels> m_input_rechannel_weights;
      Eigen::Matrix<T, head_output_channels, channels> m_head_rechannel_weights;
      Eigen::Vector<T, head_output_channels> m_head_rechannel_bias;
      wavenet_block() :
        m_input_rechannel_weights(Eigen::Matrix<T, channels, in_channels>::Constant(1.0f/24.0f)),
        m_head_rechannel_weights(Eigen::Matrix<T, head_output_channels, channels>::Constant(1.0f/24.0f)),
        m_head_rechannel_bias(Eigen::Vector<T, head_output_channels>::Zero())
      {
      }
      
      template<typename Matrix1, typename Matrix2>
      inline void process(Eigen::MatrixBase<Matrix1> const & input, Eigen::MatrixBase<Matrix2> const & bottom_input, const int n)
      {
        m_buffer.template leftCols(n).noalias() = m_input_rechannel_weights * input.template leftCols(n);
        
        // std::get<0>(m_layers).process(m_buffer, bottom_input, m_head, m_output, n);
        process_wavenet_block<layers_type, sizeof...(Layers)>::go(m_layers, m_buffer, bottom_input, m_head, m_output, n);
        
        m_head_output.template leftCols(n).noalias() = (m_head_rechannel_weights * m_head.template leftCols(n)).colwise() + m_head_rechannel_bias;
      }
    };
    
    template<
      typename T, int N,
      int in_channels, int out_channels,
      typename Block1, typename Block2
      >
    struct wavenet
    {
      typedef Eigen::Matrix<T, out_channels, N> output_type;
      
      Block1 m_block1;
      Block2 m_block2;
      template<typename Matrix>
      inline void process(Eigen::MatrixBase<Matrix> const & input, const int n)
      {
        m_block1.m_head.template leftCols(n).setZero();
        m_block1.process(input, input, n);
        m_block2.m_head.template leftCols(n) = m_block1.m_head_output.template leftCols(n);
        m_block2.process(m_block1.m_output, input, n);
      }
      const output_type &get_output()
      {
        return m_block2.m_head_output;
        // return m_block2.m_output;
      }
    };
    template<typename T, int N, int in_channels, int channels, int dilation>
    struct wavenet_direct_layer
    {
      conv1d<T, N, channels, channels, true, 1> m_conv1_1;
      Eigen::Matrix<T, channels, in_channels> m_input_mixer_weights;
      Eigen::Matrix<T, channels, channels> m_linear_weights;
      Eigen::Vector<T, channels> m_linear_bias;
    };
    
    template<typename T, int N, int in_channels, int out_channels>
    struct wavenet_direct
    {
      // Eigen::Matrix<T, 16, N> m_head1;
      
      Eigen::Matrix<T, 16, in_channels> m_rechannel1_weights;
      wavenet_direct_layer<T, N, in_channels, 16, 1> m_layer1_1;
      wavenet_direct_layer<T, N, in_channels, 16, 2> m_layer1_2;
      wavenet_direct_layer<T, N, in_channels, 16, 4> m_layer1_3;
      wavenet_direct_layer<T, N, in_channels, 16, 8> m_layer1_4;
      wavenet_direct_layer<T, N, in_channels, 16, 16> m_layer1_5;
      wavenet_direct_layer<T, N, in_channels, 16, 32> m_layer1_6;
      wavenet_direct_layer<T, N, in_channels, 16, 64> m_layer1_7;
      wavenet_direct_layer<T, N, in_channels, 16, 128> m_layer1_8;
      wavenet_direct_layer<T, N, in_channels, 16, 256> m_layer1_9;
      wavenet_direct_layer<T, N, in_channels, 16, 512> m_layer1_10;
      Eigen::Matrix<T, 8, 16> m_head_rechannel1_weights;
      template<typename Matrix>
      inline auto process(Eigen::MatrixBase<Matrix> const & input, const int n)
      {
        Eigen::Matrix<T, 16, N> head1;
        head1.setZero();
        
      }
    };
  } // namespace nam
  
} // namespace anna
