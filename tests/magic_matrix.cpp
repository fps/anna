#include <gtest/gtest.h>
#include <anna/magic.hpp>

TEST(anna, magic) {
  // auto m = anna::make_magic_matrix<float, 64, 32>();
  auto m = anna::magic_matrix_machine<float, 64, 32>();

  // ASSERT_EQ(output(0, 0), 7.0f);
}
