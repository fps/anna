#pragma once

#include <anna/conv1d.hpp>
#include <anna/log.hpp>
#include <anna/magic.hpp>
#include <anna/next_multiple.hpp>
#include <anna/tanh.hpp>

#include <Eigen/Core>

namespace anna
{
  template<typename T, int Rows, int Columns>
  auto id()
  {
    auto m = Eigen::Matrix<T, Rows, Columns>::Zero().eval();
    m.setIdentity();
    return m;
  }

  namespace op
  {
    struct empty { };

    template<typename DerivedType, typename ParametersType, typename NextOpType>
    struct chain
    {
      ParametersType m_parameters;
      NextOpType m_next_op;
  
      inline auto & end() { return m_next_op.end(); }
      inline auto & next() { return m_next_op; }
      inline auto & input() { return m_next_op.input(); }
      inline int input_head() { return m_next_op.input_head(); }

      template<int n, typename NewParametersType>
      inline void set(const NewParametersType & parameters) 
      { 
        if constexpr (0 == n)
        {
          m_parameters = parameters;
        }
        else
        {
          static_cast<DerivedType*>(this)->m_next_op.template set<n-1>(parameters);
        }
      }

      chain(const ParametersType & parameters = ParametersType()) : m_parameters(parameters) { }
    };

    template<typename T, int InputChannels, int MaxBlockSize>
    struct matrix_input
    {
      Eigen::Matrix<T, InputChannels, MaxBlockSize> m_input;
      static const int m_input_head = 0;

      inline int input_head() { return m_input_head; }
      inline auto & input() { return m_input; }
    };

    template<typename NextOpType>
    struct tanh : public chain<tanh<NextOpType>, empty, NextOpType>
    {
      typedef chain<tanh<NextOpType>, empty, NextOpType> chain_type;

      using chain_type::m_next_op;

      inline void process(const int n)
      {
        anna::inplace_eigen_fast_tanh(m_next_op.input().middleCols(m_next_op.input_head(), n));
        m_next_op.process(n);
      }
    };

    template<typename T, int Channels, int MaxBlockSize, typename NextOpType>
    struct linear1 : 
      public chain<linear1<T, Channels, MaxBlockSize, NextOpType>, Eigen::Matrix<T, Channels, Channels>, NextOpType>, 
      public matrix_input<T, Channels, MaxBlockSize>
    {
      typedef linear1<T, Channels, MaxBlockSize, NextOpType> type;
      typedef chain<type, Eigen::Matrix<T, Channels, Channels>, NextOpType> chain_type;
      typedef matrix_input<T, Channels, MaxBlockSize> matrix_input_type;

      // using chain_type::m_next_op;
      using chain_type::next;
      using chain_type::m_parameters;
      using matrix_input_type::input;
      using matrix_input_type::input_head;
  
      linear1() : chain_type(anna::id<T, Channels, Channels>()) { }
    
      inline void process(const int n)
      {
        next().input().middleCols(next().input_head(), n).noalias() = m_parameters * input().middleCols(input_head(), n);
        next().process(n);
      }
    };

    template<typename T, int OutputChannels, int InputChannels, int MaxBlockSize, typename NextOpType>
    struct linear2 : 
      public chain<linear2<T, InputChannels, OutputChannels, MaxBlockSize, NextOpType>, Eigen::Matrix<T, OutputChannels, InputChannels>, NextOpType>, 
      public matrix_input<T, InputChannels, MaxBlockSize>
    {
      typedef linear2<T, InputChannels, OutputChannels, MaxBlockSize, NextOpType> type;
      typedef chain<type, Eigen::Matrix<T, OutputChannels, InputChannels>, NextOpType> chain_type;
      typedef matrix_input<T, InputChannels, MaxBlockSize> matrix_input_type;

      using chain_type::m_parameters;
      using chain_type::m_next_op;
      using matrix_input_type::input;
      using matrix_input_type::input_head;
  
      linear2() : chain_type(anna::id<T, OutputChannels, InputChannels>()) { }
    
      inline void process(const int n)
      {
        m_next_op.input().middleCols(m_next_op.input_head(), n).noalias() = m_parameters * input().middleCols(input_head(), n);
        m_next_op.process(n);
      }
    };
   
    template<typename T, int Channels, typename NextOpType>
    struct vector_add : 
      public chain<vector_add<T, Channels, NextOpType>, Eigen::Vector<T, Channels>, NextOpType>
    {
      typedef chain<vector_add<T, Channels, NextOpType>, Eigen::Vector<T, Channels>, NextOpType> chain_type;

      using chain_type::m_next_op;
      using chain_type::m_parameters;
   
      vector_add() : chain_type(Eigen::Vector<T, Channels>::Zero()) { }

      inline void process(const int n)
      {
        m_next_op.input().middleCols(m_next_op.input_head(), n).colwise() += m_parameters;
        m_next_op.process(n);
      }
    };
    
    template<typename T, typename NextOpType>
    struct scalar_multiple : 
      public chain<scalar_multiple<T, NextOpType>, T, NextOpType>
    {
      typedef chain<scalar_multiple<T, NextOpType>, T, NextOpType> chain_type;

      using chain_type::m_next_op;
      using chain_type::m_parameters;
    
      inline void process(const int n)
      {
        m_next_op.input().middleCols(m_next_op.input_head(), n).array() *= m_parameters;
        m_next_op.process(n);
      }
    };

    template<typename T, int Channels, int MaxBlockSize>
    struct output : 
      public matrix_input<T, Channels, MaxBlockSize>
    { 
      typedef matrix_input<T, Channels, MaxBlockSize> matrix_input_type;
      
      using matrix_input_type::input;
      using matrix_input_type::input_head;

      inline auto & end() { return *this; }
      inline void process(const int n) { }
    };
    
    template<typename T, int OutputChannels, int InputChannels, int KernelSize, int Dilation, int MaxBlockSize, typename NextOpType>
    struct conv1d : 
      public chain<conv1d<T, OutputChannels, InputChannels, KernelSize, Dilation, MaxBlockSize, NextOpType>, std::array<Eigen::Matrix<T, OutputChannels, InputChannels>, KernelSize>, NextOpType>
    {
      typedef chain<conv1d<T, OutputChannels, InputChannels, KernelSize, Dilation, MaxBlockSize, NextOpType>, std::array<Eigen::Matrix<T, OutputChannels, InputChannels>, KernelSize>, NextOpType> chain_type;

      using chain_type::m_next_op;
      using chain_type::m_parameters;

      static const int magic_cols = anna::next_multiple((KernelSize - 1) * Dilation + MaxBlockSize, ANNA_PAGE_SIZE / (InputChannels * sizeof(T)));
      anna::magic_matrix_machine<T, InputChannels, magic_cols> m_magic_matrix_machine;
      Eigen::Map<Eigen::Matrix<T, InputChannels, 2 * magic_cols>> m_input;

      int m_input_head;

      conv1d() :
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

    template<typename OpType, typename InputType>
    static inline void process(OpType & op, Eigen::MatrixBase<InputType> const & input, const int n)
    {
      op.input().middleCols(op.input_head(), n) = input;
      op.process(n);
    }

    template<typename T, int OutputChannels, int InputChannels, int KernelSize, int Dilation, int MaxBlockSize, typename NextOpType>
    using conv1d_bias_tanh = conv1d<T, OutputChannels, InputChannels, KernelSize, Dilation, MaxBlockSize, vector_add<T, OutputChannels, tanh< NextOpType>>>;

    template<typename T, int OutputChannels, int InputChannels, int KernelSize, int NumDilations, int MaxBlockSize, typename NextOpType>
    struct dilated_conv1d_bias_tanh { }; 
  }
}

