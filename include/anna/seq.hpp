namespace anna
{
  template<typename Layers, int remaining>
  struct process
  {
    static void go(Layers &layers, const int n)
    {
      std::get<std::tuple_size_v<Layers> - remaining>(layers).process(std::get<std::tuple_size_v<Layers> - (remaining+1)>(layers).m_output, n);
      nn::process<Layers, remaining-1>::go(layers, n);
    }
  };
  
  template<typename Layers>
  struct process<Layers, 0>
  {
    static void go(Layers &layers, const int n)
    {

    }
  };
  
  template<typename... Layers>
  struct seq
  {
    std::tuple<Layers...> m_layers;

    template<typename Matrix>
    void process(Eigen::MatrixBase<Matrix> const & input, const int n)
    {
      std::get<0>(m_layers).process(input, n);
      
      nn::process<std::tuple<Layers...>, std::tuple_size_v<std::tuple<Layers...>> - 1>::go(m_layers, n);
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
} // namespace nn
