#include <anna/nam.hpp>
#include <iostream>
#include <chrono>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sndfile.h>

#define buffer_size 64
#define process_size 64
// #define bench_nframes (48000*100)
#define bench_nframes (64 * 4096)

int main(int argc, char *argv[])
{
  if (argc < 4)
  {
    std::cout << "Usage: " << argv[0] << " nam input output\n";
    return 1;
  }
  
  anna::nam::wavenet<
    float, buffer_size,
    1, 1,
    anna::nam::wavenet_block<
      float, buffer_size, 1, 1, 16, 8, false,
      anna::nam::wavenet_layer<float, buffer_size, 3, 1, 16, 1>,
      anna::nam::wavenet_layer<float, buffer_size, 3, 1, 16, 2>,
      anna::nam::wavenet_layer<float, buffer_size, 3, 1, 16, 4>,
      anna::nam::wavenet_layer<float, buffer_size, 3, 1, 16, 8>,
      anna::nam::wavenet_layer<float, buffer_size, 3, 1, 16, 16>,
      anna::nam::wavenet_layer<float, buffer_size, 3, 1, 16, 32>,
      anna::nam::wavenet_layer<float, buffer_size, 3, 1, 16, 64>,
      anna::nam::wavenet_layer<float, buffer_size, 3, 1, 16, 128>,
      anna::nam::wavenet_layer<float, buffer_size, 3, 1, 16, 256>,
      anna::nam::wavenet_layer<float, buffer_size, 3, 1, 16, 512>
      >,
    anna::nam::wavenet_block<
      float, buffer_size, 1, 16, 8, 1, true,
      anna::nam::wavenet_layer<float, buffer_size, 3, 1, 8, 1>,
      anna::nam::wavenet_layer<float, buffer_size, 3, 1, 8, 2>,
      anna::nam::wavenet_layer<float, buffer_size, 3, 1, 8, 4>,
      anna::nam::wavenet_layer<float, buffer_size, 3, 1, 8, 8>,
      anna::nam::wavenet_layer<float, buffer_size, 3, 1, 8, 16>,
      anna::nam::wavenet_layer<float, buffer_size, 3, 1, 8, 32>,
      anna::nam::wavenet_layer<float, buffer_size, 3, 1, 8, 64>,
      anna::nam::wavenet_layer<float, buffer_size, 3, 1, 8, 128>,
      anna::nam::wavenet_layer<float, buffer_size, 3, 1, 8, 256>,
      anna::nam::wavenet_layer<float, buffer_size, 3, 1, 8, 512, true>
      >
    > model;

  std::ifstream f("data/test.nam");
  nlohmann::json data = nlohmann::json::parse(f);
  // std::cout << data["weights"][0] << "\n";
  model.set_parameters(data["weights"].get<std::vector<float>>());

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
