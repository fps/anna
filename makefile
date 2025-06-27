EIGEN_INCLUDE ?= eigen/

CXXFLAGS = -std=c++20 -Wall -O3 -DEIGEN_STACK_ALLOCATION_LIMIT=0 -DNDEBUG -DEIGEN_NO_DEBUG -I ${EIGEN_INCLUDE}  -I include

.PHONY: all

all: build/examples/wavenet build/examples/inplace_sequence build/examples/iterated_sequence

build/examples/%: examples/%.cpp include/anna/*.hpp
	$(CXX) $(CXXFLAGS) $< -o $@
# examples/wavenet: examples/wavenet.cpp
