#include <anna/sequence.hpp>
#include <iostream>

struct f
{
  template<typename Matrix>
  auto process(Eigen::MatrixBase<Matrix> const & input, const int n)
  {
    return (const_cast<Eigen::MatrixBase<Matrix>&>(input).array() + 1).matrix();
  }
};

int main()
{
  anna::sequence::iterated::model<
    /*
    f, f, f, f, f,
    f, f, f, f, f,
    f, f, f, f, f,
    f, f, f, f, f,
    f, f, f, f, f,
    f, f, f, f, f,
    f, f, f, f, f,
    f, f, f, f, f,
    f, f, f, f, f,
    f, f, f, f, f,
    f, f, f, f, f,
    f, f, f, f, f,
    f, f, f, f, f,
    f, f, f, f, f,
    f, f, f, f, f,
    f, f, f, f, f,
    f, f, f, f, f,
    f, f, f, f, f,
    */
    f, f, f, f, f,
    f, f, f, f, f
    > model;

  Eigen::Matrix<float, 16, 32> input;
  input.setZero();

  for (int idx = 0; idx < 1000000; ++idx) {
    input = model.process(input, 32);
  }

  std::cout << input << "\n";
}
