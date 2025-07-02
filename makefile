EIGEN_INCLUDE ?= eigen/

ifeq ($(DEBUG),1)
OPT = -g3
else
OPT = -O3 -DNDEBUG -DEIGEN_NO_DEBUG 
endif

CXXFLAGS ?= -std=c++20 -Wall $(OPT) -DEIGEN_STACK_ALLOCATION_LIMIT=0 -I ${EIGEN_INCLUDE}  -I include `pkg-config nlohmann_json --cflags`

ifeq ($(COLOR),1)
CXXFLAGS += -fdiagnostics-color=always
endif

examples = nam_wavenet inplace_sequence iterated_sequence # input_output_sequence

.PHONY: all clean

all: $(addprefix build/examples/, $(examples))

build/examples/%: examples/%.cpp include/anna/*.hpp makefile
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f build/examples/*
