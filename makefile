EIGEN_INCLUDE ?= eigen/

CXXFLAGS = -std=c++20 -Wall -O3 -DEIGEN_STACK_ALLOCATION_LIMIT=0 -DNDEBUG -DEIGEN_NO_DEBUG -I ${EIGEN_INCLUDE}  -I include

examples = wavenet inplace_sequence iterated_sequence # input_output_sequence

.PHONY: all clean

all: $(addprefix build/examples/, $(examples))

build/examples/%: examples/%.cpp include/anna/*.hpp
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f build/examples/*
