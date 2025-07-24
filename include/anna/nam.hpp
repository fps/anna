#pragma once

#include <anna/conv1d.hpp>
#include <anna/tanh.hpp>

#include <cassert>
#include <iostream> // # TODO: get rid of this

namespace anna
{
  template<typename T, typename Matrix>
  void set_parameters(Eigen::MatrixBase<Matrix> const & m, std::vector<T> const & params, size_t & idx)
  {
    for (int row = 0; row < m.RowsAtCompileTime; ++row)
    {
      for (int col = 0; col < m.ColsAtCompileTime; ++col)
      {
        const_cast<Eigen::MatrixBase<Matrix>&>(m)(row, col) = params.at(idx++);
      }
    }
  }
  
  namespace nam
  {
    template<typename T, int N, int kernel_size, int bottom_in_channels, int channels, int dilation, bool last = false, bool first = false>
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

        // inplace_eigen_fast_tanh(const_cast<Eigen::MatrixBase<Matrix4>&>(output).template leftCols(n));
        const_cast<Eigen::MatrixBase<Matrix4>&>(output).template leftCols(n).noalias() += m_input_mixer_weights * bottom_input.leftCols(n);
        
        inplace_eigen_fast_tanh(const_cast<Eigen::MatrixBase<Matrix4>&>(output).template leftCols(n));
        // const_cast<Eigen::MatrixBase<Matrix4>&>(output) = output.array().tanh();
        
        if constexpr(first)
        {
          const_cast<Eigen::MatrixBase<Matrix3>&>(head).template leftCols(n).noalias() = output.template leftCols(n);
        }
        else
        {
          const_cast<Eigen::MatrixBase<Matrix3>&>(head).template leftCols(n).noalias() += output.template leftCols(n);
        }

        if constexpr(!last) {
          const_cast<Eigen::MatrixBase<Matrix4>&>(output).template leftCols(n) = (m_linear_weights * output.template leftCols(n)).colwise() + m_linear_bias;
        
          const_cast<Eigen::MatrixBase<Matrix4>&>(output).template leftCols(n).noalias() += input.template leftCols(n);
        }
      }

      void set_parameters(std::vector<T> const & params, size_t & idx)
      {
        m_dilated.set_parameters(params, idx);
        anna::set_parameters(m_input_mixer_weights, params, idx);
        anna::set_parameters(m_linear_weights, params, idx);
        anna::set_parameters(m_linear_bias, params, idx);
      }
    };
    
    template<typename T, typename Layers, int remaining>
    struct wavenet_block_set_parameters
    {
      static inline void go(Layers & layers, std::vector<T> const & params, size_t & idx)
      {
        std::get<std::tuple_size_v<Layers> - remaining>(layers).set_parameters(params, idx);
        wavenet_block_set_parameters<T, Layers, remaining-1>::go(layers, params, idx);        
      }
    }; 

    template<typename T, typename Layers>
    struct wavenet_block_set_parameters<T, Layers, 0>
    {
      static inline void go(Layers & layers, std::vector<T> const & params, size_t &idx)
      {
      
      }
    };

    template<typename Layers, int remaining>
    struct process_wavenet_block
    {
      template<typename Matrix1, typename Matrix2, typename Matrix3, typename Matrix4>
      static inline void go(Layers &layers, Eigen::MatrixBase<Matrix1> const & input, Eigen::MatrixBase<Matrix2> const & bottom_input, Eigen::MatrixBase<Matrix3> const & head, Eigen::MatrixBase<Matrix4> const & output, const int n)
      {
        std::get<std::tuple_size_v<Layers> - remaining>(layers).process(input, bottom_input, head, output, n);
        process_wavenet_block<Layers, remaining-1>::go(layers, output, bottom_input, head, input, n);
        // const_cast<Eigen::MatrixBase<Matrix1>&>(input).template leftCols(n).noalias() = output.template leftCols(n);
        // process_wavenet_block<Layers, remaining-1>::go(layers, input, bottom_input, head, output, n);
      }
    };
    
    template<typename Layers>
    struct process_wavenet_block<Layers, 0>
    {
      template<typename Matrix1, typename Matrix2, typename Matrix3, typename Matrix4>
      static inline void go(Layers &layers, Eigen::MatrixBase<Matrix1> const & input, Eigen::MatrixBase<Matrix2> const & bottom_input, Eigen::MatrixBase<Matrix3> const & head, Eigen::MatrixBase<Matrix4> const & output, const int n)
      {

      }
    };
    
    template<typename T, int N, int bottom_in_channels, int in_channels, int channels, int head_output_channels, bool head_bias, typename... Layers>
    struct wavenet_block
    {
      typedef std::tuple<Layers...> layers_type;
      layers_type m_layers;
      
      typedef Eigen::Matrix<T, channels, N> head_type;
      head_type m_head;
      
      typedef Eigen::Matrix<T, channels, N> buffer_type;
      buffer_type m_buffer1;
      buffer_type m_buffer2;
      
      typedef Eigen::Matrix<T, head_output_channels, N> head_output_type;
      head_output_type m_head_output;
      
      Eigen::Matrix<T, channels, in_channels> m_input_rechannel_weights;
      Eigen::Matrix<T, head_output_channels, channels> m_head_rechannel_weights;
      Eigen::Vector<T, head_output_channels> m_head_rechannel_bias;

      wavenet_block() :
        /*
        m_input_rechannel_weights(Eigen::Matrix<T, channels, in_channels>::Constant(1.0f/24.0f)),
        m_head_rechannel_weights(Eigen::Matrix<T, head_output_channels, channels>::Constant(1.0f/24.0f)),
        */
        m_input_rechannel_weights(Eigen::Matrix<T, channels, in_channels>::Zero()),
        m_head_rechannel_weights(Eigen::Matrix<T, head_output_channels, channels>::Zero()),
        m_head_rechannel_bias(Eigen::Vector<T, head_output_channels>::Zero())
      {

      }
      
      template<typename Matrix1, typename Matrix2>
      inline void process(Eigen::MatrixBase<Matrix1> const & input, Eigen::MatrixBase<Matrix2> const & bottom_input, const int n)
      {
        m_buffer1.template leftCols(n).noalias() = m_input_rechannel_weights * input.template leftCols(n);
        
        // std::get<0>(m_layers).process(m_buffer, bottom_input, m_head, m_output, n);
        process_wavenet_block<layers_type, sizeof...(Layers)>::go(m_layers, m_buffer1, bottom_input, m_head, m_buffer2, n);

        // TODO: Only do this conditionally when head_bias == true
        m_head_output.template leftCols(n).noalias() = (m_head_rechannel_weights * m_head.template leftCols(n)).colwise() + m_head_rechannel_bias;
      }

     
      void set_parameters(std::vector<T> const & params, size_t & idx)
      {
        anna::set_parameters(m_input_rechannel_weights, params, idx);
        wavenet_block_set_parameters<T, std::tuple<Layers...>, sizeof...(Layers)>::go(m_layers, params, idx);
        anna::set_parameters(m_head_rechannel_weights, params, idx);
        if constexpr (head_bias)
        {
          anna::set_parameters(m_head_rechannel_bias, params, idx);
        }
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

      T m_head_scale;

      wavenet() :
        m_head_scale(1.0)
      {
          
      }
      
      template<typename Matrix>
      inline void process(Eigen::MatrixBase<Matrix> const & bottom_input, const int n)
      {
        // m_block1.m_head.template leftCols(n).setZero();
        m_block1.process(bottom_input, bottom_input, n);

        m_block2.m_head.template leftCols(n) = m_block1.m_head_output.template leftCols(n);
        m_block2.process(m_block1.m_buffer1, bottom_input, n);

        m_block2.m_head_output.template leftCols(n).array() *= m_head_scale;
      }

      const output_type &get_output()
      {
        return m_block2.m_head_output;
        // return m_block2.m_output;
      }

      void set_parameters(std::vector<T> const & params)
      {
        size_t idx = 0;
        m_block1.set_parameters(params, idx);
        m_block2.set_parameters(params, idx);
        m_head_scale = params.at(idx++);
        // std::cout << idx << "\n";
        assert(idx == params.size());
      }
    };
  } // namespace nam
  
} // namespace anna
