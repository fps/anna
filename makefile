EIGEN_INCLUDE ?= eigen/

ANNA_PAGE_SIZE ?= 4096

ifeq ($(DEBUG),1)
OPT = -g3
else
OPT = -O3 -DNDEBUG -DEIGEN_NO_DEBUG
# OPT = -O3 -DNDEBUG -DEIGEN_NO_DEBUG -fno-omit-frame-pointer
# OPT = -O3 -DNDEBUG -DEIGEN_NO_DEBUG -march=native -msse4 -mfma
endif

CXXFLAGS ?= -std=gnu++20 -Wall $(OPT) -DANNA_PAGE_SIZE=${ANNA_PAGE_SIZE} -DEIGEN_STACK_ALLOCATION_LIMIT=0 -I ${EIGEN_INCLUDE}  -I include `pkg-config nlohmann_json sndfile --cflags`

ifeq ($(COLOR),1)
CXXFLAGS += -fdiagnostics-color=always
endif

LDFLAGS ?= `pkg-config sndfile --libs`

examples = ringbuffer nam_wavenet inplace_sequence iterated_sequence conv1d conv1d2by2 # input_output_sequence
benchmarks = mul conv1d

.PHONY: all clean

all: $(addprefix build/examples/, $(examples)) $(addprefix build/benchmarks/, $(benchmarks))

build/examples/%: examples/%.cpp include/anna/*.hpp makefile
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) -o $@

build/benchmarks/%: benchmarks/%.cpp include/anna/*.hpp makefile
	$(CXX) $(CXXFLAGS) `pkg-config benchmark --cflags` -fno-omit-frame-pointer $< $(LDFLAGS) `pkg-config benchmark --libs` -o $@

clean:
	rm -f build/examples/*
