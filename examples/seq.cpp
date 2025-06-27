#include <anna/anna.hpp>
#include <iostream>

#define buffer_size 64
#define process_size 8
#define bench_nframes (48000*100)
// #define bench_nframes (64 * 1000)

int main()
{
  anna::seq<
    anna::conv1d<float, buffer_size, 1, 1, 1, false, 8, 1>
    > model;
  /*
  anna::seq<
    anna::conv1d<float, buffer_size, 1, 1, 8, false, 1, 1>,
    anna::conv1d<float, buffer_size, 3, 8, 8, false, 1, 1>,
    anna::conv1d<float, buffer_size, 3, 8, 8, false, 2, 1>,
    anna::conv1d<float, buffer_size, 3, 8, 8, false, 4, 1>,
    anna::conv1d<float, buffer_size, 3, 8, 8, false, 8, 1>,
    anna::conv1d<float, buffer_size, 3, 8, 8, false, 16, 1>,
    anna::conv1d<float, buffer_size, 3, 8, 8, false, 32, 1>,
    anna::conv1d<float, buffer_size, 3, 8, 8, false, 64, 1>,
    anna::conv1d<float, buffer_size, 3, 8, 8, false, 128, 1>,
    anna::conv1d<float, buffer_size, 3, 8, 8, false, 256, 1>,
    anna::conv1d<float, buffer_size, 3, 8, 8, false, 512, 1>,
    anna::conv1d<float, buffer_size, 3, 8, 8, false, 1, 1>,
    anna::conv1d<float, buffer_size, 3, 8, 8, false, 2, 1>,
    anna::conv1d<float, buffer_size, 3, 8, 8, false, 4, 1>,
    anna::conv1d<float, buffer_size, 3, 8, 8, false, 8, 1>,
    anna::conv1d<float, buffer_size, 3, 8, 8, false, 16, 1>,
    anna::conv1d<float, buffer_size, 3, 8, 8, false, 32, 1>,
    anna::conv1d<float, buffer_size, 3, 8, 8, false, 64, 1>,
    anna::conv1d<float, buffer_size, 3, 8, 8, false, 128, 1>,
    anna::conv1d<float, buffer_size, 3, 8, 8, false, 256, 1>,
    anna::conv1d<float, buffer_size, 3, 8, 8, false, 512, 1>,
    anna::conv1d<float, buffer_size, 3, 8, 8, false, 1, 1>,
    anna::conv1d<float, buffer_size, 3, 8, 8, false, 2, 1>,
    anna::conv1d<float, buffer_size, 3, 8, 8, false, 4, 1>,
    anna::conv1d<float, buffer_size, 3, 8, 8, false, 8, 1>,
    anna::conv1d<float, buffer_size, 3, 8, 8, false, 16, 1>,
    anna::conv1d<float, buffer_size, 3, 8, 8, false, 32, 1>,
    anna::conv1d<float, buffer_size, 3, 8, 8, false, 64, 1>,
    anna::conv1d<float, buffer_size, 3, 8, 8, false, 128, 1>,
    anna::conv1d<float, buffer_size, 3, 8, 8, false, 256, 1>,
    anna::conv1d<float, buffer_size, 3, 8, 8, false, 512, 1>,
    anna::conv1d<float, buffer_size, 3, 8, 8, false, 1, 1>,
    anna::conv1d<float, buffer_size, 3, 8, 8, false, 2, 1>,
    anna::conv1d<float, buffer_size, 3, 8, 8, false, 4, 1>,
    anna::conv1d<float, buffer_size, 3, 8, 8, false, 8, 1>,
    anna::conv1d<float, buffer_size, 3, 8, 8, false, 16, 1>,
    anna::conv1d<float, buffer_size, 3, 8, 8, false, 32, 1>,
    anna::conv1d<float, buffer_size, 3, 8, 8, false, 64, 1>,
    anna::conv1d<float, buffer_size, 3, 8, 8, false, 128, 1>,
    anna::conv1d<float, buffer_size, 3, 8, 8, false, 256, 1>,
    anna::conv1d<float, buffer_size, 3, 8, 8, false, 512, 1>,
    anna::conv1d<float, buffer_size, 1, 8, 1, false, 1, 1>
    > model;
  */
  /*
  anna::seq<
    anna::conv1d<float, buffer_size, 1, 1, 16, 1, 1>,
    anna::conv1d<float, buffer_size, 3, 16, 16, 1, 1>,
    anna::conv1d<float, buffer_size, 3, 16, 16, 2, 1>,
    anna::conv1d<float, buffer_size, 3, 16, 16, 4, 1>,
    anna::conv1d<float, buffer_size, 3, 16, 16, 8, 1>,
    anna::conv1d<float, buffer_size, 3, 16, 16, 16, 1>,
    anna::conv1d<float, buffer_size, 3, 16, 16, 32, 1>,
    anna::conv1d<float, buffer_size, 3, 16, 16, 64, 1>,
    anna::conv1d<float, buffer_size, 3, 16, 16, 128, 1>,
    anna::conv1d<float, buffer_size, 3, 16, 16, 256, 1>,
    anna::conv1d<float, buffer_size, 3, 16, 16, 512, 1>,
    anna::conv1d<float, buffer_size, 1, 16, 8, 1, 1>,
    anna::conv1d<float, buffer_size, 3, 8, 8, 1, 1>,
    anna::conv1d<float, buffer_size, 3, 8, 8, 2, 1>,
    anna::conv1d<float, buffer_size, 3, 8, 8, 4, 1>,
    anna::conv1d<float, buffer_size, 3, 8, 8, 8, 1>,
    anna::conv1d<float, buffer_size, 3, 8, 8, 16, 1>,
    anna::conv1d<float, buffer_size, 3, 8, 8, 32, 1>,
    anna::conv1d<float, buffer_size, 3, 8, 8, 64, 1>,
    anna::conv1d<float, buffer_size, 3, 8, 8, 128, 1>,
    anna::conv1d<float, buffer_size, 3, 8, 8, 256, 1>,
    anna::conv1d<float, buffer_size, 3, 8, 8, 512, 1>,
    anna::conv1d<float, buffer_size, 1, 8, 1, 1, 1>
    > model;
  */
  
  // Eigen::Matrix<float, 1, buffer_size> input = Eigen::Matrix<float, 1, buffer_size>::Random();
  Eigen::Matrix<float, 1, buffer_size> input;
  input.setZero();

  Eigen::Matrix<float, 1, buffer_size> output;
  output.setZero();

    for (size_t idx = 0; idx < process_size; ++idx) {
    input(0, idx) = (1.0f + (float)idx) / 640.0f;
  }

  // std::cout << input << "\n";

  for (size_t idx = 0; idx < bench_nframes/process_size; ++idx) {
    // std::cout << idx << "\n";
    model.process(input, output, process_size);
    // std::cout << model.get_output<float, buffer_size, 1>() << "\n";    
  }

  // Eigen::Matrix<float, 1, buffer_size> output;
  // model.get_output<41, float, buffer_size, 1>(output);
  // std::cout << output << "\n";
  // std::cout << model.get_nth_output<41, float, buffer_size, 1>() << "\n";
  std::cout << model.get_output<float, buffer_size, 1>() << "\n";
}
