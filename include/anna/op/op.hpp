#pragma once

#include <anna/conv1d.hpp>
#include <anna/log.hpp>
#include <anna/next_multiple.hpp>
#include <anna/tanh.hpp>

#include <Eigen/Core>

#include <algorithm>

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
  
      inline auto & end() { return next().end(); }
      inline auto & next() { return m_next_op; }
      inline auto & input() { return next().input(); }
      inline int input_head() { return next().input_head(); }

      template<int n, typename NewParametersType>
      inline void set(const NewParametersType & parameters) 
      { 
        if constexpr (0 == n)
          {
            m_parameters = parameters;
          }
        else
          {
            static_cast<DerivedType*>(this)->next().template set<n-1>(parameters);
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

    template<typename T>
    T leaky_relu_func(T x)
    {
      return std::max((T)0.01 * x, x);
    }
    
    template<typename T, typename NextOpType>
    struct leaky_relu :
      public chain<leaky_relu<T, NextOpType>, empty, NextOpType>
    {
      typedef chain<leaky_relu<T, NextOpType>, empty, NextOpType> chain_type;

      using chain_type::next;

      inline void process(const int n)
      {
        next().input().middleCols(next().input_head(), n) = next().input().middleCols(next().input_head(), n).unaryExpr(&leaky_relu_func<T>);
        next().process(n);
      }
    };

    template<typename T, typename OpType, typename NextOpType>
    struct additive_shortcut :
      public chain<additive_shortcut<T, OpType, NextOpType>, empty, NextOpType>
    {
      using chain<additive_shortcut<T, OpType, NextOpType>, empty, NextOpType>::next;
      
      OpType m_op;
      inline auto & input() { return next().input(); }
      inline int input_head() { return next().input_head(); }

      inline void process(const int n)
      {
        m_op.input().middleCols(m_op.input_head(), n).noalias() = next().input().middleCols(next().input_head(), n);

        const int op_head = m_op.end().input_head();

        m_op.process(n);

        next().input().middleCols(next().input_head(), n).noalias() += m_op.end().input().middleCols(op_head, n);

        next().process(n);
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
      public chain<linear2<T, OutputChannels, InputChannels, MaxBlockSize, NextOpType>, Eigen::Matrix<T, OutputChannels, InputChannels>, NextOpType>, 
      public matrix_input<T, InputChannels, MaxBlockSize>
    {
      typedef linear2<T, OutputChannels, InputChannels, MaxBlockSize, NextOpType> type;
      typedef chain<type, Eigen::Matrix<T, OutputChannels, InputChannels>, NextOpType> chain_type;
      typedef matrix_input<T, InputChannels, MaxBlockSize> matrix_input_type;

      using chain_type::m_parameters;
      using chain_type::next;
      using matrix_input_type::input;
      using matrix_input_type::input_head;
  
      linear2() : chain_type(anna::id<T, OutputChannels, InputChannels>()) { }
    
      inline void process(const int n)
      {
        next().input().middleCols(next().input_head(), n).noalias() = m_parameters * input().middleCols(input_head(), n);
        next().process(n);
      }
    };
   
    template<typename T, int Channels, typename NextOpType>
    struct vector_add : 
      public chain<vector_add<T, Channels, NextOpType>, Eigen::Vector<T, Channels>, NextOpType>
    {
      typedef chain<vector_add<T, Channels, NextOpType>, Eigen::Vector<T, Channels>, NextOpType> chain_type;

      using chain_type::next;
      using chain_type::m_parameters;
   
      vector_add() : chain_type(Eigen::Vector<T, Channels>::Zero()) { }

      inline void process(const int n)
      {
        next().input().middleCols(next().input_head(), n).colwise() += m_parameters;
        next().process(n);
      }
    };
    
    template<typename T, typename NextOpType>
    struct scalar_multiple : 
      public chain<scalar_multiple<T, NextOpType>, T, NextOpType>
    {
      typedef chain<scalar_multiple<T, NextOpType>, T, NextOpType> chain_type;

      using chain_type::next;
      using chain_type::m_parameters;
    
      inline void process(const int n)
      {
        next().input().middleCols(next().input_head(), n).array() *= m_parameters;
        next().process(n);
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

    template<typename OpType, typename InputType>
    static inline void process(OpType & op, Eigen::MatrixBase<InputType> const & input, const int n)
    {
      op.input().middleCols(op.input_head(), n) = input;
      op.process(n);
    }
  }
}

