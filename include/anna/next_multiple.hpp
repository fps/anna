#pragma once

namespace anna
{
  constexpr int next_multiple(int n, int k)
  {
    return (((n-1) / k) + 1) * k;
  }
} // namespace anna
