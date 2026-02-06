#pragma once

#include <anna/conv1d.hpp>
#include <anna/make_matrix_array.hpp>
#include <anna/magic.hpp>
#include <anna/tanh.hpp>
#include <anna/next_multiple.hpp>

namespace anna
{
  namespace examples
  {
    template<typename T, int InputChannels, int Channels, int KernelSize, int Dilation, int MaxBlockSize>
    struct nam_wavenet_layer
    {
      Eigen::Matrix<T, Channels, (KernelSize-1) * Dilation + MaxBlockSize> m_dilated_inputs;
      Eigen::Matrix<T, Channels, (KernelSize-1) * Dilation + MaxBlockSize> m_dilated_outputs;

      std::array<Eigen::Matrix<T, Channels, Channels>, KernelSize> m_dilated_weights;
      Eigen::Vector<T, Channels> m_dilated_bias;

      Eigen::Matrix<T, Channels, InputChannels> m_input_mixer_weights;
      Eigen::Vector<T, Channels> m_input_mixer_bias;

      Eigen::Matrix<T, Channels, Channels> m_linear_weights;

      static const int dilation = Dilation;
    };

    template<typename T, int InputChannels, int OutputChannels, int Channels1, int KernelSize1, int Channels2, int KernelSize2, int MaxBlockSize>
    struct nam_wavenet
    {
      Eigen::Matrix<T, Channels1, InputChannels> m_input_rechannel_weights1;
      Eigen::Matrix<T, Channels1, MaxBlockSize> m_head1;

      std::tuple<
        nam_wavenet_layer<T, InputChannels, Channels1, KernelSize1, 1 << 0, MaxBlockSize>,
        nam_wavenet_layer<T, InputChannels, Channels1, KernelSize1, 1 << 1, MaxBlockSize>,
        nam_wavenet_layer<T, InputChannels, Channels1, KernelSize1, 1 << 2, MaxBlockSize>,
        nam_wavenet_layer<T, InputChannels, Channels1, KernelSize1, 1 << 3, MaxBlockSize>,
        nam_wavenet_layer<T, InputChannels, Channels1, KernelSize1, 1 << 4, MaxBlockSize>,
        nam_wavenet_layer<T, InputChannels, Channels1, KernelSize1, 1 << 5, MaxBlockSize>,
        nam_wavenet_layer<T, InputChannels, Channels1, KernelSize1, 1 << 6, MaxBlockSize>,
        nam_wavenet_layer<T, InputChannels, Channels1, KernelSize1, 1 << 7, MaxBlockSize>,
        nam_wavenet_layer<T, InputChannels, Channels1, KernelSize1, 1 << 8, MaxBlockSize>,
        nam_wavenet_layer<T, InputChannels, Channels1, KernelSize1, 1 << 9, MaxBlockSize>
      > m_layers1;
   
      Eigen::Matrix<T, Channels2, Channels1> m_head_rechannel_weights1;
      Eigen::Vector<T, Channels2> m_head_rechannel_bias1;

      Eigen::Matrix<T, Channels2, Channels1> m_input_rechannel_weights2;
      Eigen::Matrix<T, Channels2, MaxBlockSize> m_head2;
      
      std::tuple<
        nam_wavenet_layer<T, InputChannels, Channels2, KernelSize2, 1 << 0, MaxBlockSize>,
        nam_wavenet_layer<T, InputChannels, Channels2, KernelSize2, 1 << 1, MaxBlockSize>,
        nam_wavenet_layer<T, InputChannels, Channels2, KernelSize2, 1 << 2, MaxBlockSize>,
        nam_wavenet_layer<T, InputChannels, Channels2, KernelSize2, 1 << 3, MaxBlockSize>,
        nam_wavenet_layer<T, InputChannels, Channels2, KernelSize2, 1 << 4, MaxBlockSize>,
        nam_wavenet_layer<T, InputChannels, Channels2, KernelSize2, 1 << 5, MaxBlockSize>,
        nam_wavenet_layer<T, InputChannels, Channels2, KernelSize2, 1 << 6, MaxBlockSize>,
        nam_wavenet_layer<T, InputChannels, Channels2, KernelSize2, 1 << 7, MaxBlockSize>,
        nam_wavenet_layer<T, InputChannels, Channels2, KernelSize2, 1 << 8, MaxBlockSize>,
        nam_wavenet_layer<T, InputChannels, Channels2, KernelSize2, 1 << 9, MaxBlockSize>
      > m_layers2;

      T m_head_scale;

      nam_wavenet() 
      {

      }

      template<typename InputType, typename HeadType, typename LayerType>
      inline void process_layer(Eigen::MatrixBase<InputType> const & input, Eigen::MatrixBase<HeadType> const & const_head, LayerType & layer, const int & layer_idx, const int n)
      {
        Eigen::MatrixBase<HeadType> &head = const_cast<Eigen::MatrixBase<HeadType>&>(const_head);
        anna::conv1d(layer.m_dilated_weights, layer.m_dilated_bias, 1 << layer_idx, layer.m_dilated_inputs, layer.m_dilated_outputs, n, 2 * (1 << layer_idx) + n, n);

        layer.m_dilated_outputs.leftCols(n).noalias() += layer.m_input_mixer_weights * input.leftCols(n);

        anna::inplace_eigen_fast_tanh(layer.m_dilated_outputs.leftCols(n));

        layer.m_dilated_outputs.leftCols(n).noalias() += layer.m_linear_weights * layer.m_dilated_inputs.leftCols(n);

        if (0 == layer_idx)
        {
          head.leftCols(n).noalias() = layer.m_dilated_outputs.leftCols(n);
        }
        else
        {
          head.leftCols(n).noalias() += layer.m_dilated_outputs.leftCols(n);
        }
        // ++layer_idx;
      }

      template<typename InputType, typename OutputType>
      inline void process(Eigen::MatrixBase<InputType> const & input, Eigen::MatrixBase<OutputType> const & output, const int n)
      {
        std::get<0>(m_layers1).m_dilated_inputs.leftCols(n).noalias() = m_input_rechannel_weights1 * input.leftCols(n);
        /*
        int layer_idx = 0;
        process_layer(input, m_head1, std::get<0>(m_layers1), layer_idx, n);
        process_layer(input, m_head1, std::get<1>(m_layers1), layer_idx, n);
        process_layer(input, m_head1, std::get<2>(m_layers1), layer_idx, n);
        process_layer(input, m_head1, std::get<3>(m_layers1), layer_idx, n);
        process_layer(input, m_head1, std::get<4>(m_layers1), layer_idx, n);
        process_layer(input, m_head1, std::get<5>(m_layers1), layer_idx, n);
        process_layer(input, m_head1, std::get<6>(m_layers1), layer_idx, n);
        process_layer(input, m_head1, std::get<7>(m_layers1), layer_idx, n);
        process_layer(input, m_head1, std::get<8>(m_layers1), layer_idx, n);
        process_layer(input, m_head1, std::get<9>(m_layers1), layer_idx, n);

        layer_idx = 0;
        process_layer(input, m_head2, std::get<0>(m_layers2), layer_idx, n);
        process_layer(input, m_head2, std::get<1>(m_layers2), layer_idx, n);
        process_layer(input, m_head2, std::get<2>(m_layers2), layer_idx, n);
        process_layer(input, m_head2, std::get<3>(m_layers2), layer_idx, n);
        process_layer(input, m_head2, std::get<4>(m_layers2), layer_idx, n);
        process_layer(input, m_head2, std::get<5>(m_layers2), layer_idx, n);
        process_layer(input, m_head2, std::get<6>(m_layers2), layer_idx, n);
        process_layer(input, m_head2, std::get<7>(m_layers2), layer_idx, n);
        process_layer(input, m_head2, std::get<8>(m_layers2), layer_idx, n);
        process_layer(input, m_head2, std::get<9>(m_layers2), layer_idx, n);
        */
        
        int layer_idx = 0;
        std::apply([this, &input, &n, &layer_idx, &head = this->m_head1](auto&&... layer) {(this->process_layer(input, head, layer, layer_idx++, n), ...);}, m_layers1);

        layer_idx = 0;
        std::apply([this, &input, &n, &layer_idx, &head = this->m_head2](auto&&... layer) {(this->process_layer(input, head, layer, layer_idx++, n), ...);}, m_layers2);
      }  
    };
  }
} 

