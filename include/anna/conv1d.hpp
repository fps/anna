#pragma once

#include <Eigen/Core>

namespace anna
{
  template<typename WeightsType, typename InputType, typename OutputType>
  static void inline conv1d(WeightsType const & weights, const int dilation, Eigen::MatrixBase<InputType> const & input, Eigen::MatrixBase<OutputType> const & const_output, const int n, const int input_head, const int output_head)
  {
    constexpr int KernelSize = weights.size();
    Eigen::MatrixBase<OutputType> & output = const_cast<Eigen::MatrixBase<OutputType> &>(const_output);
  
    output.middleCols(output_head - n, n).noalias() = weights[0] * input.template middleCols(input_head - (KernelSize - 1  + n), n);
    for (int k = 1; k < KernelSize; ++k)
    {
      output.middleCols(output_head - n, n).noalias() += weights[k] * input.template middleCols(input_head - ((KernelSize - k - 1) * dilation + n), n);
    }
  }
  
  template<typename WeightsType, typename BiasType, typename InputType, typename OutputType>
  static void inline conv1d(WeightsType const & weights, Eigen::MatrixBase<BiasType> const & bias, const int dilation, Eigen::MatrixBase<InputType> const & input, Eigen::MatrixBase<OutputType> const & const_output, const int n, const int input_head, const int output_head)
  {
    Eigen::MatrixBase<OutputType> & output = const_cast<Eigen::MatrixBase<OutputType> &>(const_output);
  
    conv1d(weights, dilation, input, output, n, input_head, output_head);
    output.middleCols(output_head - n, n).colwise() += bias;
  }
} // namespace

