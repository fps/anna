#pragma once

#include <Eigen/Core>

#include <iostream>

namespace anna
{
  template<typename WeightsType, typename InputType, typename OutputType>
  static void inline conv1d(WeightsType const & weights, const int dilation, Eigen::MatrixBase<InputType> const & input, Eigen::MatrixBase<OutputType> const & const_output, const int n, const int input_head, const int output_head)
  {
    constexpr int KernelSize = weights.size();
    Eigen::MatrixBase<OutputType> & output = const_cast<Eigen::MatrixBase<OutputType> &>(const_output);

    output.middleCols(output_head, n).noalias() = weights[0] * input.middleCols(input_head - ((KernelSize - 1) * dilation), n);
    for (int k = 1; k < KernelSize; ++k)
    {
      output.middleCols(output_head, n).noalias() += weights[k] * input.middleCols(input_head - ((KernelSize - k - 1) * dilation), n);
    }
  }
  
  template<typename WeightsType, typename BiasType, typename InputType, typename OutputType>
  static void inline conv1d(WeightsType const & weights, Eigen::MatrixBase<BiasType> const & bias, const int dilation, Eigen::MatrixBase<InputType> const & input, Eigen::MatrixBase<OutputType> const & const_output, const int n, const int input_head, const int output_head)
  {
    conv1d(weights, dilation, input, const_output, n, input_head, output_head);

    Eigen::MatrixBase<OutputType> & output = const_cast<Eigen::MatrixBase<OutputType> &>(const_output);
    output.middleCols(output_head, n).colwise() += bias;
  }
} // namespace

