#pragma once

#include <unistd.h>
#include <sys/mman.h>
#include <cstdint>

#include <stdexcept>

namespace anna
{
  void *create_magic(int number_of_pages, const char *name)
  {
    int size = number_of_pages * getpagesize();

    int fd = memfd_create(name, 0);
    if (-1 == fd)
    {
      throw std::runtime_error("Failed to create anonymous file");
    }

    {
      int ret = ftruncate(fd, size);

      if (-1 == ret)
      {
        throw std::runtime_error("Failed to resize anonymous file");
      }
    }

    uint8_t *buffer = (uint8_t *)mmap(NULL, 2 * size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (MAP_FAILED == buffer)
    {
      throw std::runtime_error("Failed to get appropriate memory location");
    }

    {
      void *ret = mmap(buffer, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd, 0);
      if (MAP_FAILED == ret)
      {
        throw std::runtime_error("Failed to map the first time");
      }
    }

    // Now map it again, in the next virtual page
    {
      void *ret = mmap(buffer + size, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd, 0);
      if (MAP_FAILED == ret)
      {
        throw std::runtime_error("Failed to map the second time");
      }
    }

    return buffer;
  }
}
