#pragma once

#include <anna/conv1d.hpp>
#include <anna/make_matrix_array.hpp>
#include <anna/magic.hpp>
#include <anna/tanh.hpp>

namespace anna
{
  namespace examples
  {
    template<typename T, int InputChannels, int OutputChannels, int Channels1, int KernelSize1, int Channels2, int KernelSize2, int Layers, int MaxBlockSize>
    struct nam_wavenet
    {
      // #### BLOCK 1 ####
      // input rechannel for block 1
      Eigen::Matrix<T, Channels1, InputChannels> m_input_rechannel_weights1;
   
      // layers in block 1
      std::array<Eigen::Matrix<T, Channels1, MaxBlockSize>, Layers> m_dilated_inputs1;
      std::array<Eigen::Matrix<T, Channels1, MaxBlockSize>, Layers> m_dilated_outputs1;

      std::array<std::array<Eigen::Matrix<T, Channels1, Channels1>, KernelSize1>, Layers> m_dilated_weights1;
      std::array<Eigen::Vector<T, Channels1>, Layers> m_dilated_bias1;

      // input mixer for block 1
      std::array<Eigen::Matrix<T, Channels1, InputChannels>, Layers> m_input_mixer_weights1;

      Eigen::Matrix<T, Channels1, MaxBlockSize> m_head1;

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

      Eigen::Matrix<T, Channels2, MaxBlockSize> m_head2;
      
      // #### FINAL HEAD SCALE ####
      T m_head_scale;

      nam_wavenet() :
        m_input_rechannel_weights1(Eigen::Matrix<T, Channels1, InputChannels>::Ones()),
        m_dilated_inputs1({{Eigen::Matrix<T, Channels1, MaxBlockSize>::Zero()}}),
        m_dilated_outputs1({{Eigen::Matrix<T, Channels1, MaxBlockSize>::Zero()}}),
        m_dilated_weights1({{Eigen::Matrix<T, Channels1, Channels1>::Ones()}}),
        m_input_mixer_weights1({{Eigen::Matrix<T, Channels1, InputChannels>::Ones()}})
      {

      }

      template<typename InputType, typename OutputType>
      void process(Eigen::MatrixBase<InputType>  const & input, Eigen::MatrixBase<OutputType> const & output, const int n)
      {
        // block 1
        m_dilated_inputs1[0].leftCols(n).noalias() = m_input_rechannel_weights1 * input.leftCols(n);

        for (int layer = 0; layer < Layers; ++layer)
        {
          anna::conv1d(m_dilated_weights1[layer], m_dilated_bias1[layer], 1 << layer, m_dilated_inputs1[layer], m_dilated_outputs1[layer], n, MaxBlockSize, MaxBlockSize);
          m_dilated_outputs1[layer].leftCols(n).noalias() += m_input_mixer_weights1[layer] * input.leftCols(n);
          anna::inplace_eigen_fast_tanh(m_dilated_outputs1[layer].leftCols(n));

          if (0 == layer)
          {
            m_head1.leftCols(n) = m_dilated_outputs1[layer].leftCols(n);
          }
          else
          {
            m_head1.leftCols(n) += m_dilated_outputs1[layer].leftCols(n);
          }
        }

        // block 2
        m_dilated_inputs2[0].leftCols(n) = m_input_rechannel_weights2 * m_dilated_outputs1[Layers - 1].leftCols(n);

        for (int layer = 0; layer < Layers; ++layer)
        {
          anna::conv1d(m_dilated_weights2[layer], m_dilated_bias2[layer], 1 << layer, m_dilated_inputs2[layer], m_dilated_outputs2[layer], n, MaxBlockSize, MaxBlockSize);
          m_dilated_outputs2[layer].leftCols(n).noalias() += m_input_mixer_weights2[layer] * m_dilated_outputs1[Layers - 1].leftCols(n);
          anna::inplace_eigen_fast_tanh(m_dilated_outputs2[layer].leftCols(n));

          if (0 == layer)
          {
            m_head2.leftCols(n) = m_dilated_outputs2[layer].leftCols(n);
          }
          else
          {
            m_head2.leftCols(n) += m_dilated_outputs2[layer].leftCols(n);
          }
        }
      }  
    };
  }
} 

