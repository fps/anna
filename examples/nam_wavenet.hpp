#pragma once

#include <anna/conv1d.hpp>
#include <anna/make_matrix_array.hpp>
#include <anna/magic.hpp>
#include <anna/tanh.hpp>

namespace anna
{
  namespace examples
  {
    template<typename T, int InputChannels, int Channels, int KernelSize, int MaxBlockSize>
    struct nam_wavenet_layer
    {
      Eigen::Matrix<T, Channels, MaxBlockSize> m_dilated_inputs;
      Eigen::Matrix<T, Channels, MaxBlockSize> m_dilated_outputs;

      std::array<Eigen::Matrix<T, Channels, Channels>, KernelSize> m_dilated_weights;
      Eigen::Vector<T, Channels> m_dilated_bias;

      Eigen::Matrix<T, Channels, InputChannels> m_input_mixer_weights;
      Eigen::Vector<T, Channels> m_input_mixer_bias;

      Eigen::Matrix<T, Channels, Channels> m_linear_weights;
    };

    template<typename T, int InputChannels, int OutputChannels, int Channels1, int KernelSize1, int Channels2, int KernelSize2, int Layers, int MaxBlockSize>
    struct nam_wavenet
    {
      // #### BLOCK 1 ####
      // input rechannel for block 1
      Eigen::Matrix<T, Channels1, InputChannels> m_input_rechannel_weights1;

      std::array<nam_wavenet_layer<T, InputChannels, Channels1, KernelSize1, MaxBlockSize>, Layers> m_layers1;
   
      Eigen::Matrix<T, Channels1, MaxBlockSize> m_head1;

      // #### BLOCK 1 => BLOCK 2

      Eigen::Matrix<T, Channels2, Channels1> m_head_rechannel_weights1;
      Eigen::Vector<T, Channels2> m_head_rechannel_bias1;

      // #### BLOCK 2 ####
      // input rechannel for block 2
      Eigen::Matrix<T, Channels2, Channels1> m_input_rechannel_weights2;

      // layers in block 2
      std::array<Eigen::Matrix<T, Channels2, MaxBlockSize>, Layers> m_dilated_inputs2;
      std::array<Eigen::Matrix<T, Channels2, MaxBlockSize>, Layers> m_dilated_outputs2;

      std::array<std::array<Eigen::Matrix<T, Channels2, Channels2>, KernelSize2>, Layers> m_dilated_weights2;
      std::array<Eigen::Vector<T, Channels2>, Layers> m_dilated_bias2;

      // input mixer for block 2
      std::array<Eigen::Matrix<T, Channels2, Channels1>, Layers> m_input_mixer_weights2;

      std::array<Eigen::Matrix<T, Channels2, Channels2>, Layers> m_linear_weights2;

      Eigen::Matrix<T, Channels2, MaxBlockSize> m_head2;
      
      // #### FINAL HEAD SCALE ####
      T m_head_scale;

      nam_wavenet() 
      {

      }

      template<typename InputType, typename OutputType>
      inline void process(Eigen::MatrixBase<InputType>  const & input, Eigen::MatrixBase<OutputType> const & output, const int n)
      {
        // __builtin_prefetch((void*)this, 0, 0);
        // block 1
        m_layers1[0].m_dilated_inputs.leftCols(n).noalias() = m_input_rechannel_weights1 * input.leftCols(n);

        for (int layer = 0; layer < Layers; ++layer)
        {
          auto & l = m_layers1[layer];
          anna::conv1d(l.m_dilated_weights, l.m_dilated_bias, 1 << layer, l.m_dilated_inputs, l.m_dilated_outputs, n, 2 * (1 << layer) + n, n);

          l.m_dilated_outputs.leftCols(n).noalias() += l.m_input_mixer_weights * input.leftCols(n);

          anna::inplace_eigen_fast_tanh(l.m_dilated_outputs.leftCols(n));

          l.m_dilated_outputs.leftCols(n).noalias() += l.m_linear_weights * l.m_dilated_inputs.leftCols(n);

          if (0 == layer)
          {
            m_head1.leftCols(n).noalias() = l.m_dilated_outputs.leftCols(n);
          }
          else
          {
            m_head1.leftCols(n).noalias() += l.m_dilated_outputs.leftCols(n);
          }
        }
        

        m_head2.leftCols(n).noalias() = m_head_rechannel_weights1 * m_head1.leftCols(n);
        m_head2.leftCols(n).colwise() += m_head_rechannel_bias1;

        // block 2
        m_dilated_inputs2[0].leftCols(n).noalias() = m_input_rechannel_weights2 * m_layers1[Layers-1].m_dilated_outputs.leftCols(n);

        for (int layer = 0; layer < Layers; ++layer)
        {
          anna::conv1d(m_dilated_weights2[layer], m_dilated_bias2[layer], 1 << layer, m_dilated_inputs2[layer], m_dilated_outputs2[layer], n, 2 * (1 << layer) + n, n);

          m_dilated_outputs2[layer].leftCols(n).noalias() += m_input_mixer_weights2[layer] * m_layers1[Layers-1].m_dilated_outputs.leftCols(n);

          anna::inplace_eigen_fast_tanh(m_dilated_outputs2[layer].leftCols(n));

          m_dilated_outputs2[layer].leftCols(n).noalias() += m_linear_weights2[layer] * m_dilated_inputs2[layer].leftCols(n);

          m_head2.leftCols(n).noalias() += m_dilated_outputs2[layer].leftCols(n);
        }
      }  
    };
  }
} 

