#pragma once

#include <unistd.h>
#include <sys/mman.h>
#include <cstdint>

#include <stdexcept>
#include <Eigen/Core>

#include <iostream>

namespace anna
{
  static void *create_magic(const int number_of_pages, const int number_of_mirrored_pages, const char *name)
  {
    if (number_of_mirrored_pages > number_of_pages)
    {
      throw std::runtime_error("number_of_mirrored_pages > number_of_pages");
    }

    int ps = getpagesize();

    int fd = memfd_create(name, 0);
    if (-1 == fd)
    {
      throw std::runtime_error("Failed to create anonymous file");
    }

    {
      int ret = ftruncate(fd, number_of_pages * ps);

      if (-1 == ret)
      {
        throw std::runtime_error("Failed to resize anonymous file");
      }
    }

    // Allocate a continuous memory region where we can map the parts of the memfd
    void *buffer = mmap(NULL, number_of_pages * ps + number_of_mirrored_pages * ps, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (MAP_FAILED == buffer)
    {
      throw std::runtime_error("Failed to get appropriate memory location");
    }

    // Now use the known address for buffer from the previous mmap to map our
    // anonyous file from memfd_create to the first half of buffer.
    {
      void *ret = mmap(buffer, number_of_pages * ps, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd, 0);
      if ((MAP_FAILED == ret) || (ret != buffer))
      {
        throw std::runtime_error("Failed to map the first time");
      }
    }

    // And now to the second half of the buffer. The original mmap at buffer
    // is thus unmapped and we do not have to unmap it again.
    {
      void *ret = mmap(buffer + number_of_pages * ps, number_of_mirrored_pages * ps, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd, 0);
      if ((MAP_FAILED == ret) || (ret != (buffer + number_of_pages * ps)))
      {
        throw std::runtime_error("Failed to map the second time");
      }
    }

    {
      int ret = close(fd);
      if (0 != ret)
      {
        throw std::runtime_error("Failed to close file descriptor");
      }
    }

    return buffer;
  }

  static void check_pagesize()
  {
      if (getpagesize() != ANNA_PAGE_SIZE)
      {
        throw std::runtime_error("getpagesize() != ANNA_PAGE_SIZE");
      }
  }

  template<typename T, int rows, int cols>
  struct magic_matrix_machine
  {
    int m_pagesize;
    int m_number_of_pages;
    void * m_buffer;

    magic_matrix_machine() 
    {
      check_pagesize();

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

      m_buffer = create_magic(m_number_of_pages, m_number_of_pages, "anna-magic-matrix");
    }

    Eigen::Map<Eigen::Matrix<T, rows, 2*cols>> get_map()
    {
      return Eigen::Map<Eigen::Matrix<T, rows, 2*cols>>((T*)m_buffer);
    }

    ~magic_matrix_machine()
    {
      // std::cout << "TODO: ~magic_matrix_machine: cleanup ;)" << "\n";
      int ret = munmap(m_buffer, m_pagesize * m_number_of_pages * 2);
      if (0 != ret)
      {
        // TODO: What to do here? ;)
      }
    }
  };
}
