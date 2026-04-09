#pragma once

#include <anna/conv1d.hpp>
#include <anna/make_matrix_array.hpp>
#include <anna/magic.hpp>
#include <anna/tanh.hpp>
#include <anna/next_multiple.hpp>
#include <anna/log.hpp>

#include <sstream>

namespace anna
{
  namespace examples
  {
    template<typename MatrixType, typename ParametersIterator>
    void set_matrix_entries(Eigen::MatrixBase<MatrixType> const & const_m, ParametersIterator & it)
    {
      Eigen::MatrixBase<MatrixType> & m = const_cast<Eigen::MatrixBase<MatrixType>&>(const_m);
      for (int row = 0; row < m.rows(); ++row)
        for (int col = 0; col < m.cols(); ++col)
          m(row, col) = *(it++);
    }

    template<typename T, int InputChannels, int Channels, int KernelSize, int Dilation, int MaxBlockSize>
    struct nam_wavenet_layer
    {
      static const int magic_cols = anna::next_multiple((KernelSize - 1) * Dilation + MaxBlockSize, ANNA_PAGE_SIZE / (Channels * sizeof(T)));
      anna::magic_matrix_machine<T, Channels, magic_cols> m_magic_matrix_machine;
      Eigen::Map<Eigen::Matrix<T, Channels, 2 * magic_cols>> m_input;

      std::array<Eigen::Matrix<T, Channels, Channels>, KernelSize> m_dilated_weights;
      Eigen::Vector<T, Channels> m_dilated_bias;

      Eigen::Matrix<T, Channels, InputChannels> m_input_mixer_weights;

      Eigen::Matrix<T, Channels, Channels> m_linear_weights;
      Eigen::Vector<T, Channels> m_linear_bias;

      int m_input_head;
      static const int m_dilation = Dilation;

      static const int m_max_block_size = MaxBlockSize;

      inline void advance_head(const int n)
      {
        DBG("advance by " << n << ". before: " << m_input_head)
        m_input_head += n;
        if ((m_input_head + m_max_block_size) >= 2 * magic_cols)
          m_input_head -= magic_cols;

        // m_input_head = (m_input_head + m_max_block_size - magic_cols) % magic_cols + magic_cols - m_max_block_size;
        DBG("after: " << m_input_head)
      }

      nam_wavenet_layer() :
        m_input(m_magic_matrix_machine.get_map()),
        m_input_head(magic_cols)
      {
        DBG("Channels: " << Channels << ". Dilation: " << Dilation << ", magic_cols: " << magic_cols << ", pages: " << ((m_input.rows() * m_input.cols() * sizeof(T)) / ANNA_PAGE_SIZE))
      }

      template<typename ParametersIterator>
      void set_parameters(ParametersIterator &it)
      {
        for (int row = 0; row < m_dilated_weights[0].rows(); ++row)
          for (int col = 0; col < m_dilated_weights[0].cols(); ++col)
            for (size_t k = 0; k < m_dilated_weights.size(); ++k)
              m_dilated_weights[k](row, col) = *(it++);
              // m_dilated_weights[m_dilated_weights.size() - k - 1](row, col) = *(it++);

        set_matrix_entries(m_dilated_bias, it);

        set_matrix_entries(m_input_mixer_weights, it);

        set_matrix_entries(m_linear_weights, it);
        set_matrix_entries(m_linear_bias, it);
      }   
    };

    template<typename T, int Channels, int MaxBlockSize>
    struct output
    {
      Eigen::Matrix<T, Channels, MaxBlockSize> m_input;
      static const int m_input_head = 0;

      inline void advance_head(const int n)
      {
        // NO OP
      }
    };

    template<typename T, int InputChannels, int OutputChannels, int Channels1, int KernelSize1, int Channels2, int KernelSize2, int MaxBlockSize>
    struct nam_wavenet
    {
      Eigen::Matrix<T, Channels1, InputChannels> m_rechannel_weights1;
      Eigen::Matrix<T, Channels1, MaxBlockSize> m_head1;

      nam_wavenet_layer<T, InputChannels, Channels1, KernelSize1, 1 << 0, MaxBlockSize> m_layer10;
      nam_wavenet_layer<T, InputChannels, Channels1, KernelSize1, 1 << 1, MaxBlockSize> m_layer11;
      nam_wavenet_layer<T, InputChannels, Channels1, KernelSize1, 1 << 2, MaxBlockSize> m_layer12;
      nam_wavenet_layer<T, InputChannels, Channels1, KernelSize1, 1 << 3, MaxBlockSize> m_layer13;
      nam_wavenet_layer<T, InputChannels, Channels1, KernelSize1, 1 << 4, MaxBlockSize> m_layer14;
      nam_wavenet_layer<T, InputChannels, Channels1, KernelSize1, 1 << 5, MaxBlockSize> m_layer15;
      nam_wavenet_layer<T, InputChannels, Channels1, KernelSize1, 1 << 6, MaxBlockSize> m_layer16;
      nam_wavenet_layer<T, InputChannels, Channels1, KernelSize1, 1 << 7, MaxBlockSize> m_layer17;
      nam_wavenet_layer<T, InputChannels, Channels1, KernelSize1, 1 << 8, MaxBlockSize> m_layer18;
      nam_wavenet_layer<T, InputChannels, Channels1, KernelSize1, 1 << 9, MaxBlockSize> m_layer19;
      output<T, Channels1, MaxBlockSize> m_layer1a;
 
      Eigen::Matrix<T, Channels2, Channels1> m_head_rechannels_weights1;

      Eigen::Matrix<T, Channels2, Channels1> m_rechannel_weights2;
      Eigen::Matrix<T, Channels2, MaxBlockSize> m_head2;
      
      nam_wavenet_layer<T, InputChannels, Channels2, KernelSize2, 1 << 0, MaxBlockSize> m_layer20;
      nam_wavenet_layer<T, InputChannels, Channels2, KernelSize2, 1 << 1, MaxBlockSize> m_layer21;
      nam_wavenet_layer<T, InputChannels, Channels2, KernelSize2, 1 << 2, MaxBlockSize> m_layer22;
      nam_wavenet_layer<T, InputChannels, Channels2, KernelSize2, 1 << 3, MaxBlockSize> m_layer23;
      nam_wavenet_layer<T, InputChannels, Channels2, KernelSize2, 1 << 4, MaxBlockSize> m_layer24;
      nam_wavenet_layer<T, InputChannels, Channels2, KernelSize2, 1 << 5, MaxBlockSize> m_layer25;
      nam_wavenet_layer<T, InputChannels, Channels2, KernelSize2, 1 << 6, MaxBlockSize> m_layer26;
      nam_wavenet_layer<T, InputChannels, Channels2, KernelSize2, 1 << 7, MaxBlockSize> m_layer27;
      nam_wavenet_layer<T, InputChannels, Channels2, KernelSize2, 1 << 8, MaxBlockSize> m_layer28;
      nam_wavenet_layer<T, InputChannels, Channels2, KernelSize2, 1 << 9, MaxBlockSize> m_layer29;
      output<T, Channels2, MaxBlockSize> m_layer2a;

      Eigen::Matrix<T, OutputChannels, Channels2> m_head_rechannel_weights2;
      Eigen::Vector<T, OutputChannels> m_head_rechannel_bias2;

      T m_head_scale;

      nam_wavenet() 
      {

      }

      template<typename InputType, typename HeadType, typename LayerType, typename NextLayerType>
      inline void process_layer(Eigen::MatrixBase<InputType> const & input, Eigen::MatrixBase<HeadType> const & const_head, LayerType & layer, NextLayerType & next_layer, const bool first, const int n)
      {
        DBG("process_layer")
        Eigen::MatrixBase<HeadType> &head = const_cast<Eigen::MatrixBase<HeadType>&>(const_head);
        anna::conv1d(layer.m_dilated_weights, layer.m_dilated_bias, layer.m_dilation, layer.m_input, next_layer.m_input, n, layer.m_input_head, next_layer.m_input_head);

        next_layer.m_input.middleCols(next_layer.m_input_head, n).noalias() += layer.m_input_mixer_weights * input.leftCols(n);

        anna::inplace_eigen_fast_tanh(next_layer.m_input.middleCols(next_layer.m_input_head, n));

        next_layer.m_input.middleCols(next_layer.m_input_head, n).noalias() += layer.m_linear_weights * layer.m_input.middleCols(layer.m_input_head, n);
        next_layer.m_input.middleCols(next_layer.m_input_head, n).colwise() += layer.m_linear_bias;

        if (true == first)
        {
          head.leftCols(n).noalias() = next_layer.m_input.middleCols(next_layer.m_input_head, n);
        }
        else
        {
          head.leftCols(n).noalias() += next_layer.m_input.middleCols(next_layer.m_input_head, n);
        }

        layer.advance_head(n);
      }

      template<typename InputType, typename OutputType>
      inline void process(Eigen::MatrixBase<InputType> const & input, Eigen::MatrixBase<OutputType> const & output, const int n)
      {
        DBG("process")
        DBG("m_layer10.m_input_head, n: " << m_layer10.m_input_head << ", " << n)
        m_layer10.m_input.middleCols(m_layer10.m_input_head, n).noalias() = m_rechannel_weights1 * input.leftCols(n);

        process_layer(input, m_head1, m_layer10, m_layer11, true, n);
        process_layer(input, m_head1, m_layer11, m_layer12, false, n);
        process_layer(input, m_head1, m_layer12, m_layer13, false, n);
        process_layer(input, m_head1, m_layer13, m_layer14, false, n);
        process_layer(input, m_head1, m_layer14, m_layer15, false, n);
        process_layer(input, m_head1, m_layer15, m_layer16, false, n);
        process_layer(input, m_head1, m_layer16, m_layer17, false, n);
        process_layer(input, m_head1, m_layer17, m_layer18, false, n);
        process_layer(input, m_head1, m_layer18, m_layer19, false, n);
        process_layer(input, m_head1, m_layer19, m_layer1a, false, n);
       
        m_head2.leftCols(n).noalias() = m_head_rechannels_weights1 * m_head1.leftCols(n);

        m_layer20.m_input.middleCols(m_layer20.m_input_head, n).noalias() = m_rechannel_weights2 * m_layer1a.m_input.middleCols(m_layer1a.m_input_head, n);

        process_layer(input, m_head2, m_layer20, m_layer21, false, n);
        process_layer(input, m_head2, m_layer21, m_layer22, false, n);
        process_layer(input, m_head2, m_layer22, m_layer23, false, n);
        process_layer(input, m_head2, m_layer23, m_layer24, false, n);
        process_layer(input, m_head2, m_layer24, m_layer25, false, n);
        process_layer(input, m_head2, m_layer25, m_layer26, false, n);
        process_layer(input, m_head2, m_layer26, m_layer27, false, n);
        process_layer(input, m_head2, m_layer27, m_layer28, false, n);
        process_layer(input, m_head2, m_layer28, m_layer29, false, n);
        process_layer(input, m_head2, m_layer29, m_layer2a, false, n);

        const_cast<Eigen::MatrixBase<OutputType>&>(output).leftCols(n).noalias() = m_head_rechannel_weights2 * m_head2.leftCols(n);
        const_cast<Eigen::MatrixBase<OutputType>&>(output).leftCols(n).colwise() += m_head_rechannel_bias2;

        const_cast<Eigen::MatrixBase<OutputType>&>(output).leftCols(n).array() *= m_head_scale;
      }  

      template<typename ParametersIterator, typename ParametersIteratorEnd>
      void set_parameters(ParametersIterator &it, ParametersIteratorEnd & it_end)
      {
        set_matrix_entries(m_rechannel_weights1, it);

        m_layer10.set_parameters(it);
        m_layer11.set_parameters(it);
        m_layer12.set_parameters(it);
        m_layer13.set_parameters(it);
        m_layer14.set_parameters(it);
        m_layer15.set_parameters(it);
        m_layer16.set_parameters(it);
        m_layer17.set_parameters(it);
        m_layer18.set_parameters(it);
        m_layer19.set_parameters(it);

        set_matrix_entries(m_head_rechannels_weights1, it);

        set_matrix_entries(m_rechannel_weights2, it);

        m_layer20.set_parameters(it);
        m_layer21.set_parameters(it);
        m_layer22.set_parameters(it);
        m_layer23.set_parameters(it);
        m_layer24.set_parameters(it);
        m_layer25.set_parameters(it);
        m_layer26.set_parameters(it);
        m_layer27.set_parameters(it);
        m_layer28.set_parameters(it);
        m_layer29.set_parameters(it);

        set_matrix_entries(m_head_rechannel_weights2, it);
        set_matrix_entries(m_head_rechannel_bias2, it);

        m_head_scale = *(it++);
        
        if (it != it_end)
        {
          std::stringstream str;
          str << "Failed to consume the precise number of parameters: " << (it_end - it) << " remaining.";
          throw std::runtime_error(str.str());
        }
      }
    };
  }
} 

