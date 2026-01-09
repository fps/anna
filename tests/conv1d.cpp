#include <gtest/gtest.h>
#include <anna/conv1d.hpp>

TEST(anna, conv1d) {
  anna::conv1d<float, 64, 3, 2, 3, true, 4> c;
  std::vector<float> params(2*3*3+3);
  std::fill(params.begin(), params.end(), 1.0f);

  size_t idx = 0;
  c.set_parameters(params, idx);

  Eigen::Matrix<float, 2, 64*3> input = Eigen::Matrix<float, 2, 64*3>::Ones();
  Eigen::Matrix<float, 3, 64> output = Eigen::Matrix<float, 3, 64>::Zero();

  c.process(input, output, 64*3, 64, 64);

  ASSERT_EQ(output(0, 0), 7.0f);
}
