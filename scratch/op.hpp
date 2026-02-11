#pragma once

#include <anna/conv1d.hpp>
#include <anna/log.hpp>
#include <anna/magic.hpp>
#include <anna/next_multiple.hpp>
#include <anna/tanh.hpp>

#include <Eigen/Core>

namespace anna
{
  namespace op
  {
    template<typename T, int OutputChannels, int InputChannels, int KernelSize, int Dilation, int MaxBlockSize, typename NextOpType>
    struct conv1d
    {
      std::array<Eigen::Matrix<T, OutputChannels, InputChannels>, KernelSize> m_weights;

      static const int magic_cols = anna::next_multiple((KernelSize - 1) * Dilation + MaxBlockSize, ANNA_PAGE_SIZE / (InputChannels * sizeof(T)));
      anna::magic_matrix_machine<T, InputChannels, magic_cols> m_magic_matrix_machine;
      Eigen::Map<Eigen::Matrix<T, InputChannels, 2 * magic_cols>> m_input;

      int m_input_head;

      NextOpType m_next_op;

      conv1d() :
        m_input(m_magic_matrix_machine.get_map()),
        m_input_head(magic_cols)
      {
        DBG("magic_cols: " << magic_cols)
      }

      template<int n, typename ValueType>
      inline void set(ValueType value)
      {
        if constexpr (0 == n)
        {
          m_weights = value;
        }
        else
        {
          m_next_op.template set<n-1>(value);
        }
      }  
    
      inline auto & end() { return m_next_op.end(); }
    
      inline auto & input() { return m_input; }
    
      inline int input_head() { return m_input_head; }
    
      inline void process(const int n)
      {
        // m_next_op.input().middleCols(m_next_op.input_head(), n).noalias() = m_matrix * m_input.middleCols(m_input_head, n);
        anna::conv1d(m_weights, Dilation, m_input, m_next_op.input(), n, m_input_head, m_next_op.input_head());
        m_next_op.process(n);
      }
    };

    template<typename NextOpType>
    struct tanh
    {
      NextOpType m_next_op;
      
      template<int n, typename ValueType>
      inline void set(ValueType value)
      {
        if constexpr (0 == n)
        {
          ERR("tanh has no parameters")
        }
        else
        {
          m_next_op.template set<n-1>(value);
        }
      }  

      inline auto & end() { return m_next_op.end(); }
    
      inline auto & input() { return m_next_op.input(); }
    
      inline int input_head() { return m_next_op.input_head(); }
    
      inline void process(const int n)
      {
        anna::inplace_eigen_fast_tanh(m_next_op.input().middleCols(m_next_op.input_head(), n));
        m_next_op.process(n);
      }
    };

    template<typename T, int Channels, int MaxBlockSize, typename NextOpType>
    struct linear1
    {
      Eigen::Matrix<T, Channels, MaxBlockSize> m_input;
      Eigen::Matrix<T, Channels, Channels> m_matrix;
      static const int m_input_head = 0;
    
      NextOpType m_next_op;
    
      linear1() :
        m_matrix(Eigen::Matrix<T, Channels, Channels>::Zero())
      {
        for (int index = 0; index < Channels; ++index)
        {
          m_matrix(index, index) = 1;
        }
      }
    
      template<int n, typename ValueType>
      inline void set(ValueType value)
      {
        if constexpr (0 == n)
        {
          m_matrix = value;
        }
        else
        {
          m_next_op.template set<n-1>(value);
        }
      }  
    
      inline auto & end() { return m_next_op.end(); }
    
      inline auto & input() { return m_input; }
    
      inline int input_head() { return m_input_head; }
    
      inline void process(const int n)
      {
        m_next_op.input().middleCols(m_next_op.input_head(), n).noalias() = m_matrix * m_input.middleCols(m_input_head, n);
        m_next_op.process(n);
      }
    };
    
    template<typename T, int OutputChannels, int InputChannels, int MaxBlockSize, typename NextOpType>
    struct linear2
    {
      Eigen::Matrix<T, InputChannels, MaxBlockSize> m_input;
      Eigen::Matrix<T, OutputChannels, InputChannels> m_matrix;
      static const int m_input_head = 0;
    
      NextOpType m_next_op;
    
      linear2() :
        m_matrix(Eigen::Matrix<T, OutputChannels, InputChannels>::Zero())
      {
        for (int index = 0; index < std::min(InputChannels, OutputChannels); ++index)
        {
          m_matrix(index, index) = 1;
        }
      }
    
      template<int n, typename ValueType>
      inline void set(ValueType value)
      {
        if constexpr (0 == n)
        {
          m_matrix = value;
        }
        else
        {
          m_next_op.template set<n - 1>(value);
        }
      }  
    
      inline auto & end() { return m_next_op.end(); }
    
      inline auto & input() { return m_input; }
    
      inline int input_head() { return m_input_head; }
    
      inline void process(const int n)
      {
        m_next_op.input().middleCols(m_next_op.input_head(), n).noalias() = m_matrix * m_input.middleCols(m_input_head, n);
        m_next_op.process(n);
      }
    };
    
    
    template<typename T, int Channels, typename NextOpType>
    struct vector_add
    {
      Eigen::Vector<T, Channels> m_vector;
    
      NextOpType m_next_op;
    
      template<int n, typename ValueType>
      inline void set(ValueType value)
      {
        if constexpr (0 == n)
        {
          m_vector = value;
        }
        else
        {
          m_next_op.template set<n - 1>(value);
        }
      }  
    
      inline auto & end() { return m_next_op.end(); }
    
      inline auto & input() { return m_next_op.input(); }
    
      inline int input_head() { return m_next_op.input_head(); }
    
      inline void process(const int n)
      {
        m_next_op.input().middleCols(m_next_op.input_head(), n).colwise() += m_vector;
        m_next_op.process(n);
      }
    };
    
    template<typename T, int InputChannels, int MaxBlockSize>
    struct output
    {
      Eigen::Matrix<T, InputChannels, MaxBlockSize> m_input;
      static const int m_input_head = 0;
    
      template<int n, typename ValueType>
      inline void set(ValueType value)
      {
        if constexpr (0 == n)
        {
          ERR("output has no value to set()")
        }
      }  
    
      inline auto & end() { return *this; }
    
      inline auto & input() { return m_input; }
    
      inline int input_head() { return m_input_head; }
    
      inline void process(const int n) { /* NO OP */ }
    };
    
    template<typename T, int Nominator, int Denominator, typename NextOpType>
    struct scalar_multiple
    {
      NextOpType m_next_op;
    
      T m_value = (T)Nominator/(T)Denominator;
    
      scalar_multiple() :
        m_value((T)Nominator/(T)Denominator)
      {
    
      }
    
      template<int n, typename ValueType>
      inline void set(ValueType value)
      {
        if constexpr (0 == n)
        {
          m_value = value;
        }
        else
        {
          m_next_op.template set<n-1>(value);
        }
      }  
    
      inline auto & end() { return m_next_op.end(); }
    
      inline auto & input() { return m_next_op.input(); }
    
      inline int input_head() { return m_next_op.input_head(); }
    
      inline void process(const int n)
      {
        m_next_op.input().middleCols(m_next_op.input_head(), n).array() *= m_value;
        m_next_op.process(n);
      }
    };

    template<typename OpType, typename InputType>
    static inline void process(OpType & op, Eigen::MatrixBase<InputType> const & input, const int n)
    {
      op.input().middleCols(op.input_head(), n) = input;
      op.process(n);
    }
  }
}


