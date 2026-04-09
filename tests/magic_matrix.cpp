#include <gtest/gtest.h>
#include <anna/magic.hpp>

TEST(magic, ThrowOnBadRows) {
  EXPECT_ANY_THROW(({ anna::magic_matrix_machine<float, 63, 32>(); }));
}

TEST(magic, ThrowOnBadCols) {
  EXPECT_ANY_THROW(({ anna::magic_matrix_machine<float, 64, 33>(); }));
}

TEST(magic, WrapAround) {
  auto magic = anna::magic_matrix_machine<float, 64, 192>();
  auto map = magic.get_map();

  map = Eigen::Matrix<float, 64, 192*2>::Zero();
  map.template leftCols(192) = Eigen::Matrix<float, 64, 192>::Ones();

  EXPECT_EQ((map.template rightCols(192)), (Eigen::Matrix<float, 64, 192>::Ones()));
  EXPECT_NE((map.template rightCols(192)), (Eigen::Matrix<float, 64, 192>::Zero()));

  map.template rightCols(192) = Eigen::Matrix<float, 64, 192>::Zero();

  EXPECT_EQ((map.template leftCols(192)), (Eigen::Matrix<float, 64, 192>::Zero()));
  EXPECT_NE((map.template leftCols(192)), (Eigen::Matrix<float, 64, 192>::Ones()));
}
