#include <gtest/gtest.h>
#include <anna/magic.hpp>

typedef anna::magic_matrix_machine<float, 63, 32> throwing_machine1;
typedef anna::magic_matrix_machine<float, 64, 33> throwing_machine2;

TEST(magic, ThrowOnBadRows) {
  EXPECT_ANY_THROW({ throwing_machine1(); });
}

TEST(magic, ThrowOnBadCols) {
  EXPECT_ANY_THROW({ throwing_machine2(); });
}

TEST(magic, WrapAround) {
  auto magic = anna::magic_matrix_machine<float, 64, 128>();
  auto map = magic.get_map();

  map = Eigen::Matrix<float, 64, 256>::Zero();
  map.template leftCols(128) = Eigen::Matrix<float, 64, 128>::Ones();

  EXPECT_EQ((map.template rightCols(128)), (Eigen::Matrix<float, 64, 128>::Ones()));
  EXPECT_NE((map.template rightCols(128)), (Eigen::Matrix<float, 64, 128>::Zero()));

  /*
  bool equal = map.template rightCols(128) == Eigen::Matrix<float, 64, 128>::Ones();
  EXPECT_TRUE(equal);
  */
}
