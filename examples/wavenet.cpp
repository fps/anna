#include <nn/nn.hpp>
#include <iostream>
#include <chrono>

#define buffer_size 64
#define process_size 64
// #define bench_nframes (48000*100)
#define bench_nframes (64 * 4096)

int main()
{
  nn::wavenet<
    float, buffer_size,
    1, 1,
    nn::wavenet_block<
      float, buffer_size, 1, 1, 16, 8,
      nn::wavenet_layer<float, buffer_size, 3, 1, 16, 1>,
      nn::wavenet_layer<float, buffer_size, 3, 1, 16, 2>,
      nn::wavenet_layer<float, buffer_size, 3, 1, 16, 4>,
      nn::wavenet_layer<float, buffer_size, 3, 1, 16, 8>,
      nn::wavenet_layer<float, buffer_size, 3, 1, 16, 16>,
      nn::wavenet_layer<float, buffer_size, 3, 1, 16, 32>,
      nn::wavenet_layer<float, buffer_size, 3, 1, 16, 64>,
      nn::wavenet_layer<float, buffer_size, 3, 1, 16, 128>,
      nn::wavenet_layer<float, buffer_size, 3, 1, 16, 256>,
      nn::wavenet_layer<float, buffer_size, 3, 1, 16, 512>
      >,
    nn::wavenet_block<
      float, buffer_size, 1, 16, 8, 1,
      nn::wavenet_layer<float, buffer_size, 3, 1, 8, 1>,
      nn::wavenet_layer<float, buffer_size, 3, 1, 8, 2>,
      nn::wavenet_layer<float, buffer_size, 3, 1, 8, 4>,
      nn::wavenet_layer<float, buffer_size, 3, 1, 8, 8>,
      nn::wavenet_layer<float, buffer_size, 3, 1, 8, 16>,
      nn::wavenet_layer<float, buffer_size, 3, 1, 8, 32>,
      nn::wavenet_layer<float, buffer_size, 3, 1, 8, 64>,
      nn::wavenet_layer<float, buffer_size, 3, 1, 8, 128>,
      nn::wavenet_layer<float, buffer_size, 3, 1, 8, 256>,
      nn::wavenet_layer<float, buffer_size, 3, 1, 8, 512>
      >
    > model;

  Eigen::Matrix<float, 1, buffer_size> input = Eigen::Matrix<float, 1, buffer_size>::Zero();

  for (size_t idx = 0; idx < process_size; ++idx) {
    input(0, idx) = (1.0f + (float)idx) / 640.0f;
  }

  auto tick = std::chrono::high_resolution_clock::now();
  
  for (size_t idx = 0; idx < bench_nframes/process_size; ++idx) {
    model.process(input, process_size);
  }

  auto tock = std::chrono::high_resolution_clock::now();

  std::cout << (tock - tick).count() << "\n";
  std::cout << model.get_output() << "\n";
}
