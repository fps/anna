  // Taken from: https://github.com/mikeoliphant/NeuralAudio/blob/main/NeuralAudio/Activation.h
  struct FastMath
  {
    template <typename Matrix>
    static auto Tanh(Matrix& x)
    {
      float* data = x.data();
      size_t size = x.rows() * x.cols();

      for (size_t pos = 0; pos < size; pos++)
        {
          data[pos] = Tanh(data[pos]);
        }

      return x;
    }

    static inline float Tanh(const float x)
    {
      const float ax = fabsf(x);

      const float x2 = x * x;

      return (x * (2.45550750702956f + 2.45550750702956f * ax + (0.893229853513558f + 0.821226666969744f * ax) * x2)
              / (2.44506634652299f + (2.44506634652299f + x2) * fabsf(x + 0.814642734961073f * x * ax)));
    }

    static inline float Sigmoid(float x)
    {
      return  0.5f * (Tanh(x * 0.5f) + 1);
    }
  };

  inline float fast_tanh(const float &x)
  {
    const float ax = fabsf(x);
    const float x2 = x * x;
    
    return (x * (2.45550750702956f + 2.45550750702956f * ax + (0.893229853513558f + 0.821226666969744f * ax) * x2)
            / (2.44506634652299f + (2.44506634652299f + x2) * fabsf(x + 0.814642734961073f * x * ax)));
  }
  
  template<typename Matrix>
  void inplace_fast_tanh(Matrix &m)
  {
    float * const data = m.data();
    size_t size = m.RowsAtCompileTime * m.ColsAtCompileTime;
    
    for (size_t pos = 0; pos < size; ++pos) {
      data[pos] = fast_tanh(data[pos]);
    }
  }

