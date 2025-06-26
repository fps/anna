#include <anna/seq.hpp>
#include <iostream>

struct f
{
  template<typename Matrix>
  void process_inplace(const Eigen::MatrixBase<Matrix> & input, const int n)
  {
    const_cast<Eigen::MatrixBase<Matrix>&>(input).array() += 1;
  }
};

int main()
{
  anna::sequence::inplace::model<
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
    f, f, f, f, f,
    f, f, f, f, f
    > model;

  Eigen::Matrix<float, 16, 32> m;
  m.setZero();

  model.process(m, 32);

  std::cout << m << "\n";
}
