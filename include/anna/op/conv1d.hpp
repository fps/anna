#include <anna/op/op.hpp>

namespace anna
{
  namespace op
  {

    template<typename T, int OutputChannels, int InputChannels, int KernelSize, int Dilation, int MaxBlockSize, typename NextOpType>
    struct conv1d : 
      public chain<conv1d<T, OutputChannels, InputChannels, KernelSize, Dilation, MaxBlockSize, NextOpType>, std::array<Eigen::Matrix<T, OutputChannels, InputChannels>, KernelSize>, NextOpType>
    {
      typedef chain<conv1d<T, OutputChannels, InputChannels, KernelSize, Dilation, MaxBlockSize, NextOpType>, std::array<Eigen::Matrix<T, OutputChannels, InputChannels>, KernelSize>, NextOpType> chain_type;
      
      using chain_type::m_next_op;
      using chain_type::m_parameters;
      
      static const int receptive_field_size = (KernelSize - 1) * Dilation;
      static const int buffer_size = 2 * receptive_field_size + MaxBlockSize;
      
      Eigen::Matrix<T, InputChannels, buffer_size> m_input1;
      int m_input_head1;
      
      Eigen::Matrix<T, InputChannels, buffer_size> m_input2;
      int m_input_head2;
      
      conv1d() :
        m_input1(Eigen::Matrix<T, InputChannels, buffer_size>::Zero()),
        m_input_head1(receptive_field_size),
        m_input2(Eigen::Matrix<T, InputChannels, buffer_size>::Zero()),
        m_input_head2(0)
      { }
    
      inline auto & input() { return m_input1; }
      inline int input_head() { return m_input_head1; }

      inline void advance_head(int & head, const int n, Eigen::Matrix<T, InputChannels, buffer_size> & input)
      {
        head += n;
        const int overshoot = head - (2 * receptive_field_size);

        if (overshoot > 0)  {
          input.middleCols(0, overshoot) = input.middleCols(2 * receptive_field_size, overshoot);
        }
        
        head %= (2 * receptive_field_size);
      }
      
      inline void process(const int n)
      {
        // DBG("input_head: " << m_input_head1)
        m_input2.middleCols(m_input_head2, n).noalias() = m_input1.middleCols(m_input_head1, n);        
        
        if (m_input_head1 >= receptive_field_size) {
          anna::conv1d(m_parameters, Dilation, m_input1, m_next_op.input(), n, m_input_head1, m_next_op.input_head());
        }
        else {
          anna::conv1d(m_parameters, Dilation, m_input2, m_next_op.input(), n, m_input_head2, m_next_op.input_head());
        }
        
        advance_head(m_input_head1, n, m_input1);
        advance_head(m_input_head2, n, m_input2);
        
        m_next_op.process(n);
      }
    };

    template<typename T, int OutputChannels, int InputChannels, int KernelSize, int Dilation, int MaxBlockSize, typename NextOpType>
    using conv1d_bias_tanh = conv1d<T, OutputChannels, InputChannels, KernelSize, Dilation, MaxBlockSize, vector_add<T, OutputChannels, tanh<NextOpType>>>;

    template<typename T, int OutputChannels, int InputChannels, int KernelSize, int Dilation, int MaxBlockSize, typename NextOpType>
    using conv1d_bias_leaky_relu = conv1d<T, OutputChannels, InputChannels, KernelSize, Dilation, MaxBlockSize, vector_add<T, OutputChannels, leaky_relu<T, NextOpType>>>;
  }
}
