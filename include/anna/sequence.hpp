#include <Eigen/Core>

namespace anna
{
  /*
    template<typename Layers, int remaining>
    struct process
    {
    template<typename Matrix, typename Matrix2>    
    static void go(Layers &layers, Eigen::MatrixBase<Matrix> const & input, Eigen::MatrixBase<Matrix2> const & output, const int n)
    {
    std::get<std::tuple_size_v<Layers> - remaining>(layers).process(std::get<std::tuple_size_v<Layers> - (remaining+1)>(layers).m_output, n);
    anna::process<Layers, remaining-1>::go(layers, n);
    }
    };
  
    template<typename Layers>
    struct process<Layers, 0>
    {
    template<typename Matrix, typename Matrix2>    
    static void go(Layers &layers, Eigen::MatrixBase<Matrix> const & input,
    {

    }
    };
  
    template<typename... Layers>
    struct seq
    {
    std::tuple<Layers...> m_layers;

    template<typename Matrix, typename Matrix2>
    void process(Eigen::MatrixBase<Matrix> const & input, Eigen::MatrixBase<Matrix2> const & output, const int n)
    {
    std::get<0>(m_layers).process(input, output, n);
      
    anna::process<std::tuple<Layers...>, std::tuple_size_v<std::tuple<Layers...>> - 1>::go(m_layers, input, output,n);
    }

    template<int index, typename T, int N, int channels>
    const Eigen::Matrix<T, channels, N> &get_nth_output()
    {
    return std::get<index>(m_layers).m_output;
    }

    template<typename T, int N, int channels>
    const Eigen::Matrix<T, channels, N> &get_output()
    {
    return std::get<std::tuple_size_v<std::tuple<Layers...>> - 1>(m_layers).m_output;
    }
    };
  */

  namespace sequence
  {
    namespace input_output
    {
      template<typename Layers, int remaining>
      struct processor
      {
        template<typename Matrix, typename Matrix2>
        static void process(Layers & layers, Eigen::MatrixBase<Matrix> const &input, Eigen::MatrixBase<Matrix2> const &output, const int n)
        {
          std::get<std::tuple_size_v<Layers> - remaining>(layers).process_inplace(input, output, n);
          input = output;
          processor<Layers, remaining - 1>::process(layers, input, output, n);
        }
      };
    
      template<typename Layers>
      struct processor<Layers, 0>
      {
        template<typename Matrix, typename Matrix2>
        static void process(Layers & layers, Eigen::MatrixBase<Matrix> const &input, Eigen::MatrixBase<Matrix2> const &output, const int n)
        {
          
        }
      };
    
      template<typename... Layers>
      struct model
      {
        std::tuple<Layers...> m_layers;

        template<typename Matrix, typename Matrix2>
        void process(Eigen::MatrixBase<Matrix> const & input, Eigen::MatrixBase<Matrix2> const & output, const int n)
        {
          processor<std::tuple<Layers...>, std::tuple_size_v<std::tuple<Layers...>>>::process(m_layers, input, output, n);
        }
      };
    } // namespace binary


    namespace iterated
    {
      template<typename Layers, int remaining>
      struct processor
      {
        template<typename Matrix>
        static auto process(Layers &layers, Eigen::MatrixBase<Matrix> const &input, const int n)
        {
          return std::get<std::tuple_size_v<Layers> - remaining>(layers).process(processor<Layers, remaining - 1>::process(layers, input, n), n);
        }
      };
    
      template<typename Layers>
      struct processor<Layers, 1>
      {
        template<typename Matrix>
        static auto process(Layers & layers, Eigen::MatrixBase<Matrix> const &input, const int n)
        {
          return std::get<std::tuple_size_v<Layers> - 1>(layers).process(input, n);
        }
      };
    
      template<typename... Layers>
      struct model
      {
        std::tuple<Layers...> m_layers;

        template<typename M>
        auto process(Eigen::MatrixBase<M> const & input, const int n)
        {
          return processor<std::tuple<Layers...>, std::tuple_size_v<std::tuple<Layers...>>>::process(m_layers, input, n);
        }
      };
    }
    
    namespace unary
    {
      template<typename... Layers>
      struct model
      {
        
      };
    }
    
    namespace inplace
    {
      template<typename Layers, int remaining>
      struct processor
      {
        template<typename Matrix>
        static void process(Layers &layers, Eigen::MatrixBase<Matrix> const &input, const int n)
        {
          std::get<std::tuple_size_v<Layers> - remaining>(layers).process_inplace(input, n);
          processor<Layers, remaining - 1>::process(layers, input, n);
        }
      };
    
      template<typename Layers>
      struct processor<Layers, 0>
      {
        template<typename Matrix>
        static void process(Layers & layers, Eigen::MatrixBase<Matrix> const &input, const int n)
        {
          
        }
      };
    
      template<typename... Layers>
      struct model
      {
        std::tuple<Layers...> m_layers;

        template<typename Matrix>
        void process(Eigen::MatrixBase<Matrix> const & input, const int n)
        {
          processor<std::tuple<Layers...>, std::tuple_size_v<std::tuple<Layers...>>>::process(m_layers, input, n);
        }
      };
    } // namespace inplace
  } // namespace sequence
} // namespace anna
