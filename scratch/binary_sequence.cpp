#include <anna/seq.hpp>
#include <anna/conv1d.hpp>
#include <iostream>

int main()
{
  anna::sequence::input_output::model<
    anna::conv1d<float, 32, 1, 8>,
    anna::conv1d<float, 32, 8, 8>,
    anna::conv1d<float, 32, 8, 1>
    > model;

  Eigen::Matrix<float, 1, 32> input;
    Eigen::Matrix<float, 1, 32> output;
  m.setZero();

  model.process(input, output, 32);

  std::cout << m << "\n";
}
