#include <gtest/gtest.h>
#include <anna/next_multiple.hpp>
#include <Eigen/Core>

TEST(next_multiple, Simple)
{
  EXPECT_EQ(anna::next_multiple(101, 10), 110);
  EXPECT_EQ(anna::next_multiple(4111, 4096), 8192);
  EXPECT_EQ(anna::next_multiple(17, 4096), 4096);
}

TEST(next_multiple, EigenMatrix)
{
  Eigen::Matrix<float, anna::next_multiple(101, 10), anna::next_multiple(4111, 4096)> m;
  EXPECT_EQ(m.RowsAtCompileTime, 110);
}
