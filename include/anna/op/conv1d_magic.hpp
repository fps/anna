#include <anna/op/op.hpp>
#include <anna/magic.hpp>

namespace anna
{
  namespace op
  {
    template<typename T, int OutputChannels, int InputChannels, int KernelSize, int Dilation, int MaxBlockSize, typename NextOpType>
    struct conv1d_magic : 
      public chain<conv1d_magic<T, OutputChannels, InputChannels, KernelSize, Dilation, MaxBlockSize, NextOpType>, std::array<Eigen::Matrix<T, OutputChannels, InputChannels>, KernelSize>, NextOpType>
    {
      typedef chain<conv1d<T, OutputChannels, InputChannels, KernelSize, Dilation, MaxBlockSize, NextOpType>, std::array<Eigen::Matrix<T, OutputChannels, InputChannels>, KernelSize>, NextOpType> chain_type;

      using chain_type::m_next_op;
      using chain_type::m_parameters;

      static const int magic_cols = anna::next_multiple((KernelSize - 1) * Dilation + MaxBlockSize, ANNA_PAGE_SIZE / (InputChannels * sizeof(T)));
      anna::magic_matrix_machine<T, InputChannels, magic_cols> m_magic_matrix_machine;
      Eigen::Map<Eigen::Matrix<T, InputChannels, 2 * magic_cols>> m_input;

      int m_input_head;

      conv1d_magic() :
        m_input(m_magic_matrix_machine.get_map()),
        m_input_head((KernelSize - 1) * Dilation)
      {
        DBG("magic_cols: " << magic_cols)
          }

      inline auto & input() { return m_input; }
      inline int input_head() { return m_input_head; }
    
      inline void process(const int n)
      {
        DBG("input_head: " << m_input_head)
          anna::conv1d(m_parameters, Dilation, m_input, m_next_op.input(), n, m_input_head, m_next_op.input_head());
        m_input_head += n;
        if (m_input_head % magic_cols >= (KernelSize - 1) * Dilation)
          {
            m_input_head %= magic_cols;
          }
        m_next_op.process(n);
      }
    };
  }
}

