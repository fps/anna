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
    // Curiously Recurring Template Pattern
    template<typename DerivedType, typename NextOpType>
    struct crtp
    {
      NextOpType m_next_op;
  
      inline auto & end() { return m_next_op.end(); }
      inline auto & next() { return m_next_op; }
      inline auto & input() { return m_next_op.input(); }
      inline int input_head() { return m_next_op.input_head(); }
    };

    template<typename T, int InputChannels, int MaxBlockSize>
    struct matrix_input
    {
      Eigen::Matrix<T, InputChannels, MaxBlockSize> m_input;

      static const int m_input_head = 0;
      inline int input_head() { return m_input_head; }
      inline auto & input() { return m_input; }
    };

    template<typename DerivedType, typename ParametersType>
    struct parameters
    {
      ParametersType m_parameters;

      parameters(const ParametersType & parameters = ParametersType()) : m_parameters(parameters) { }

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

      auto const & get() { return m_parameters; }
    };

    template<typename DerivedType>
    struct no_parameters
    {
      template<int n, typename NewValueType>
      inline void set(const NewValueType & parameters) 
      { 
        if constexpr (0 == n)
        {
          ERR("no_parameters has no parameters")
        }
        else
        {
          static_cast<DerivedType*>(this)->m_next_op.template set<n-1>(parameters);
        }
      }
    };

    template<typename NextOpType>
    struct tanh : public crtp<tanh<NextOpType>, NextOpType>, public no_parameters<tanh<NextOpType>>
    {
      typedef crtp<tanh<NextOpType>, NextOpType> crtp_type;

      inline void process(const int n)
      {
        anna::inplace_eigen_fast_tanh(m_next_op.input().middleCols(m_next_op.input_head(), n));
        m_next_op.process(n);
      }

      using crtp_type::m_next_op;
    };

    template<typename T, int Channels, int MaxBlockSize, typename NextOpType>
    struct linear1 : 
      public crtp<linear1<T, Channels, MaxBlockSize, NextOpType>, NextOpType>, 
      public parameters<linear1<T, Channels, MaxBlockSize, NextOpType>, Eigen::Matrix<float, Channels, Channels>>,
      public matrix_input<T, Channels, MaxBlockSize>
    {
      typedef linear1<T, Channels, MaxBlockSize, NextOpType> type;
      typedef crtp<type, NextOpType> crtp_type;
      typedef parameters<type, Eigen::Matrix<float, Channels, Channels>> parameters_type;
      typedef matrix_input<float, Channels, MaxBlockSize> matrix_input_type;

      using parameters_type::m_parameters;
      using crtp_type::m_next_op;
      using matrix_input_type::input;
      using matrix_input_type::input_head;
  
      linear1() : parameters_type(anna::id<T, Channels, Channels>()) { }
    
      inline void process(const int n)
      {
        m_next_op.input().middleCols(m_next_op.input_head(), n).noalias() = m_parameters * input().middleCols(input_head(), n);
        m_next_op.process(n);
      }
    };

    template<typename T, int InputChannels, int OutputChannels, int MaxBlockSize, typename NextOpType>
    struct linear2 : 
      public crtp<linear2<T, InputChannels, OutputChannels, MaxBlockSize, NextOpType>, NextOpType>, 
      public parameters<linear2<T, InputChannels, OutputChannels, MaxBlockSize, NextOpType>, Eigen::Matrix<float, OutputChannels, InputChannels>>,
      public matrix_input<T, InputChannels, MaxBlockSize>
    {
      typedef linear2<T, InputChannels, OutputChannels, MaxBlockSize, NextOpType> type;
      typedef crtp<type, NextOpType> crtp_type;
      typedef parameters<type, Eigen::Matrix<float, OutputChannels, InputChannels>> parameters_type;
      typedef matrix_input<float, InputChannels, MaxBlockSize> matrix_input_type;

      using parameters_type::m_parameters;
      using crtp_type::m_next_op;
      using matrix_input_type::input;
      using matrix_input_type::input_head;
  
      linear2() : parameters_type(anna::id<T, OutputChannels, InputChannels>()) { }
    
      inline void process(const int n)
      {
        m_next_op.input().middleCols(m_next_op.input_head(), n).noalias() = m_parameters * input().middleCols(input_head(), n);
        m_next_op.process(n);
      }
    };
   
    template<typename T, int Channels, typename NextOpType>
    struct vector_add
    {
      Eigen::Vector<T, Channels> m_vector;
    
      NextOpType m_next_op;
    
      template<int n, typename ValueType>
      inline void set(const ValueType & value)
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
    
      inline auto & next() { return m_next_op; }

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
      inline void set(const ValueType & value)
      {
        if constexpr (0 == n)
        {
          ERR("output has no parameters to set()")
        }
      }  
    
      inline auto & next() { return *this; }

      inline auto & end() { return *this; }
    
      inline auto & input() { return m_input; }
    
      inline int input_head() { return m_input_head; }
    
      inline void process(const int n) { /* NO OP */ }
    };
    
    template<typename T, typename NextOpType>
    struct scalar_multiple
    {
      NextOpType m_next_op;
    
      T m_value;
    
      scalar_multiple() :
        m_value(T())
      {
    
      }
    
      template<int n, typename ValueType>
      inline void set(const ValueType & value)
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
    
      inline auto & next() { return m_next_op; }

      inline auto & end() { return m_next_op.end(); }
    
      inline auto & input() { return m_next_op.input(); }
    
      inline int input_head() { return m_next_op.input_head(); }
    
      inline void process(const int n)
      {
        m_next_op.input().middleCols(m_next_op.input_head(), n).array() *= m_value;
        m_next_op.process(n);
      }
    };

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
        m_input_head((KernelSize - 1) * Dilation)
      {
        DBG("magic_cols: " << magic_cols)
      }

      template<int n, typename ValueType>
      inline void set(const ValueType & value)
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
    
      inline auto & next() { return m_next_op; }

      inline auto & end() { return m_next_op.end(); }
    
      inline auto & input() { return m_input; }
    
      inline int input_head() { return m_input_head; }
    
      inline void process(const int n)
      {
        DBG("input_head: " << m_input_head)
        anna::conv1d(m_weights, Dilation, m_input, m_next_op.input(), n, m_input_head, m_next_op.input_head());
        m_input_head += n;
        if (m_input_head % magic_cols >= (KernelSize - 1) * Dilation)
        {
          m_input_head %= magic_cols;
        }
        m_next_op.process(n);
      }
    };

    template<typename T, int OutputChannels, int InputChannels, int KernelSize, int Dilation, int MaxBlockSize, typename NextOpType>
    using conv1d_bias_tanh = conv1d<T, OutputChannels, InputChannels, KernelSize, Dilation, MaxBlockSize, vector_add<T, OutputChannels, tanh< NextOpType>>>;

    template<typename T, int OutputChannels, int InputChannels, int KernelSize, int NumDilations, int MaxBlockSize, typename NextOpType>
    struct dilated_conv1d_bias_tanh { }; 

    template<typename OpType, typename InputType>
    static inline void process(OpType & op, Eigen::MatrixBase<InputType> const & input, const int n)
    {
      op.input().middleCols(op.input_head(), n) = input;
      op.process(n);
    }
  }
}


