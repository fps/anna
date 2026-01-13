#pragma once

#include <unistd.h>
#include <sys/mman.h>
#include <cstdint>

#include <stdexcept>
#include <Eigen/Core>

namespace anna
{
  static void *create_magic(int number_of_pages, const char *name)
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

  template<typename T, int rows, int cols>
  struct magic_matrix_machine
  {
    int m_pagesize;
    int m_number_of_pages;
    void * m_buffer;

    magic_matrix_machine() 
    {
      m_pagesize = getpagesize();
      // std::cout << "pagesize: " << m_pagesize << "\n";
      if (m_pagesize % (sizeof(T) * rows) != 0)
      {
        throw std::runtime_error("(sizeof(T) * rows) not a divisor of pagesize");
      }

      if ((sizeof(T) * rows * cols) % m_pagesize != 0)
      {
        throw std::runtime_error("pagesize not a divisor of (sizeof(T) * rows * cols)");
      }

      m_number_of_pages = (sizeof(T) * rows * cols) / m_pagesize;

      m_buffer = create_magic(m_number_of_pages, "anna-magic-matrix");
    }

    Eigen::Map<Eigen::Matrix<T, rows, 2*cols>> get_map()
    {
      return Eigen::Map<Eigen::Matrix<T, rows, 2*cols>>((T*)m_buffer);
    }

    ~magic_matrix_machine()
    {
      std::cout << "TODO: ~magic_matrix_machine: cleanup ;)" << "\n";
    }

  };

  template<typename T, int rows, int cols>
  Eigen::Map<Eigen::Matrix<T, rows, cols>> make_magic_matrix()
  {
    
  }
}
