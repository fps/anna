EIGEN_INCLUDE ?= eigen/

ifeq ($(DEBUG),1)
OPT = -g3
else
OPT = -O3 -DNDEBUG -DEIGEN_NO_DEBUG 
endif

CXXFLAGS ?= -std=c++20 -Wall $(OPT) -DEIGEN_STACK_ALLOCATION_LIMIT=0 -I ${EIGEN_INCLUDE}  -I include `pkg-config nlohmann_json sndfile --cflags`

ifeq ($(COLOR),1)
CXXFLAGS += -fdiagnostics-color=always
endif

LDFLAGS ?= `pkg-config sndfile --libs`

examples = nam_wavenet inplace_sequence iterated_sequence conv1d # input_output_sequence

.PHONY: all clean

all: $(addprefix build/examples/, $(examples))

build/examples/%: examples/%.cpp include/anna/*.hpp makefile
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) -o $@

clean:
	rm -f build/examples/*
