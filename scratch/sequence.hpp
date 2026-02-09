#include <Eigen/Core>

namespace anna
{
  namespace sequence
  {
    namespace input_output
    {
      template<typename Layers, int remaining>
      struct processor
      {
        template<typename Matrix>
        static inline void process(Layers & layers, Eigen::MatrixBase<Matrix> const &input, const int n)
        {
          std::get<std::tuple_size_v<Layers> - remaining>(layers).process_cached(input, n);
          // input = output;
          return processor<Layers, remaining - 1>::process(layers, std::get<std::tuple_size_v<Layers> - remaining>(layers).m_output, n);
        }
      };
    
      template<typename Layers>
      struct processor<Layers, 0>
      {
        template<typename Matrix>
        static inline void process(Layers & layers, Eigen::MatrixBase<Matrix> const &input, const int n)
        {
          // return const_cast<Eigen::MatrixBase<Matrix>&>(input); 
          // return input; 
        }
      };
    
      template<typename... Layers>
      struct model
      {
        std::tuple<Layers...> m_layers;

        template<typename Matrix>
        inline void process(Eigen::MatrixBase<Matrix> const & input, const int n)
        {
          std::get<0>(m_layers).process_cached(input, n);
          processor<std::tuple<Layers...>, std::tuple_size_v<std::tuple<Layers...>>-1>::process(m_layers, std::get<0>(m_layers).m_output, n);
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
