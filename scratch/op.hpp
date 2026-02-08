#pragma once

namespace anna
{
  template<typename InputType, typename MType, typename NextOpType>
  struct linear
  {
    static inline void operator()(Eigen::MatrixBase<InputType> const & input, Eigen::MatrixBase<M> const & m, NextOpType next_op)
    {
    
    }
  };
}
