#pragma once

#include <iostream>
#include <sstream>
#include <stdexcept>

#ifdef NDEBUG
  #define DBG(x) { }
#else
  #define DBG(x) { std::cout << "DEBUG: " << x << std::endl; }
#endif

#define ERR(x) { std::cout << "ERROR: " << x << "\n"; std::stringstream ss; ss << "ERROR: " << x; throw std::runtime_error(ss.str().c_str()); }
#define INFO(x) { std::cout << "INFO:  " << x << "\n";}

