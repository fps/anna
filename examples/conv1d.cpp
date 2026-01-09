#include <anna/conv1d_double_buffer.hpp>
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>

#define buffer_size 50

int main()
{
  std::ifstream f("julia/conv1d.json");
  nlohmann::json data = nlohmann::json::parse(f);

  anna::conv1d<float, buffer_size, 3, 16, 16, true, 13> conv1d;

  auto parameters = data["parameters"].get<std::vector<float>>();

  size_t pidx = 0;
  conv1d.set_parameters(parameters, pidx);

  assert(pidx == parameters.size());

  auto test_input = data["input"].get<std::vector<float>>();
  auto test_output = data["output"].get<std::vector<float>>();

  auto input_mat = Eigen::Map<Eigen::Matrix<float, 16, 100>>(test_input.data());
  
  std::vector<float> output(1600, 0);
  auto output_mat = Eigen::Map<Eigen::Matrix<float, 16, 100>>(output.data());
  
  for (size_t chunk = 0; chunk < 100/buffer_size; ++chunk)
  {
    conv1d.process(input_mat.middleCols(chunk*buffer_size, buffer_size), output_mat.middleCols(chunk*buffer_size, buffer_size), buffer_size);
  }

  for (size_t idx = 0; idx < 100; ++idx)
  {
    std::cout << std::setprecision(9) << test_input.at(test_input.size() - 100 + idx) << " => (" << test_output.at(test_output.size() - 100 + idx) << " - " << output.at(output.size() - 100 + idx) << ") = " << test_output.at(test_output.size() - 100 + idx) - output.at(output.size() - 100 + idx) << "\n";
  }

  // std::cout << output_mat.transpose() << "\n";
}
