#include <anna/magic.hpp>
#include <iostream>
#include <cassert>

#include <Eigen/Core>

int main(int argc, char *argv[])
{
  int pagesize = getpagesize();

  assert(pagesize == ANNA_PAGE_SIZE);

  int8_t * magic_buffer = (int8_t*)anna::create_magic(1, 1, "lalala");

  magic_buffer[pagesize] = 87;

  std::cout << "pagesize: " << pagesize << " ... " << (int)magic_buffer[0] << "\n";

  Eigen::Map<Eigen::Matrix<float, 16, 2*ANNA_PAGE_SIZE/(16*sizeof(float))>, Eigen::Aligned> magic_matrix((float*)magic_buffer);

  magic_matrix(0, 64) = 1;

  std::cout << magic_matrix.transpose() << "\n";
}
