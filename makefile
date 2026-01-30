EIGEN_INCLUDE ?= eigen-5.0.0/

ANNA_PAGE_SIZE ?= $(shell getconf PAGE_SIZE)

ARCHFLAGS ?= -march=native

ifeq ($(DEBUG),1)
OPTFLAGS ?= -g -O1 -fno-omit-frame-pointer
else
OPTFLAGS ?= -O3 -DNDEBUG -DEIGEN_NO_DEBUG -fno-trapping-math -fno-math-errno
endif

CXXFLAGS ?= -std=gnu++20 -Wall -Wdouble-promotion $(ARCHFLAGS) $(OPTFLAGS) -DANNA_PAGE_SIZE=${ANNA_PAGE_SIZE} -DEIGEN_STACK_ALLOCATION_LIMIT=0 -I ${EIGEN_INCLUDE}  -I include `pkg-config nlohmann_json sndfile --cflags` -I vendored/vectorclass2

ifeq ($(COLOR),1)
CXXFLAGS += -fdiagnostics-color=always
endif

LDFLAGS ?= `pkg-config sndfile --libs` 

examples = 
benchmarks = conv1d conv1d_fixed_buffersize conv1d_function
tests = conv1d magic_matrix next_multiple

.PHONY: all clean tests benchmarks examples check

all: examples benchmarks tests 

tests: $(addprefix build/tests/, $(tests))

check: tests
	for n in build/tests/*; do "$$n"; done

benchmarks: $(addprefix build/benchmarks/, $(benchmarks))

examples: $(addprefix build/examples/, $(examples))

build/examples/%: examples/%.cpp include/anna/*.hpp
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) -o $@

build/benchmarks/%: benchmarks/%.cpp include/anna/*.hpp
	$(CXX) $(CXXFLAGS) `pkg-config benchmark --cflags` $< $(LDFLAGS) `pkg-config benchmark --libs` -o $@

build/tests/%: tests/%.cpp include/anna/*.hpp
	$(CXX) $(CXXFLAGS) `pkg-config gtest_main --cflags` $< `pkg-config gtest_main --libs` -o $@

clean:
	rm -f build/examples/*
