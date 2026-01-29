#include <gtest/gtest.h>

#include <anna/conv1d.hpp>
#include <anna/make_matrix_array.hpp>

TEST(anna, conv1d) 
{
  Eigen::Matrix<float, 2, 64*3> input = Eigen::Matrix<float, 2, 64*3>::Ones();
  Eigen::Matrix<float, 3, 64> output = Eigen::Matrix<float, 3, 64>::Zero();

  std::array<Eigen::Matrix<float, 3, 2>, 3> weights = anna::make_matrix_array<float, 3, 2, 3>(1.0f);
  Eigen::Vector<float, 3> bias = Eigen::Vector<float, 3>::Ones();

  anna::conv1d<3, 4>::process_with_bias(weights, bias, input, output, 64, 64*3, 64);

  ASSERT_EQ(output(0, 0), 7.0f);
}

TEST(anna, conv1d_receptive_field1) 
{
  ASSERT_EQ((anna::conv1d<3, 1024>::receptive_field), 2048);
}
