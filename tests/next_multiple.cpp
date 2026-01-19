#include <gtest/gtest.h>
#include <anna/next_multiple.hpp>

TEST(next_multiple, all)
{
  EXPECT_EQ(anna::next_multiple(101, 10), 110);
  EXPECT_EQ(anna::next_multiple(4111, 4096), 8192);
}
