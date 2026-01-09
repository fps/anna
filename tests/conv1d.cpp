#include <gtest/gtest.h>
#include <anna/conv1d.hpp>

TEST(anna, conv1d) {
  anna::conv1d<float, 3, 2, 2, true, 4> c;
  ASSERT_EQ(1, 1);
}
