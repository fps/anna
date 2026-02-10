#pragma once

#include <iostream>

#ifdef NDEBUG
  #define DBG(x) { }
#else
  #define DBG(x) { std::cout << "DEBUG: " << x << std::endl; }
#endif

#define ERR(x) { std::cout << "ERROR: " << x << "\n";}
#define INFO(x) { std::cout << "INFO:  " << x << "\n";}

