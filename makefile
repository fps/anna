EIGEN_INCLUDE ?= eigen/

ANNA_PAGE_SIZE ?= $(shell getconf PAGE_SIZE)

ARCHFLAGS ?= -march=native

ifeq ($(DEBUG),1)
# OPTFLAGS = -g3
OPTFLAGS ?= -g -O1
else
OPTFLAGS ?= -O3 -DNDEBUG -DEIGEN_NO_DEBUG -fno-trapping-math -fno-math-errno
# OPTFLAGS = -O3 -DNDEBUG -DEIGEN_NO_DEBUG -mavx2 -fabi-version=0 -mfma
# OPTFLAGS = -O3 -DNDEBUG -DEIGEN_NO_DEBUG -fno-omit-frame-pointer
# OPTFLAGS = -O3 -DNDEBUG -DEIGEN_NO_DEBUG -march=native -msse4 -mfma
endif

# CXXFLAGS ?= -std=gnu++20 -Wall $(OPTFLAGS) -DEIGEN_DONT_VECTORIZE -DEIGEN_RUNTIME_NO_MALLOC -DANNA_PAGE_SIZE=${ANNA_PAGE_SIZE} -DEIGEN_STACK_ALLOCATION_LIMIT=0 -I ${EIGEN_INCLUDE}  -I include `pkg-config nlohmann_json sndfile --cflags`
CXXFLAGS ?= -std=gnu++20 -Wall -Wdouble-promotion $(ARCHFLAGS) $(OPTFLAGS) -DANNA_PAGE_SIZE=${ANNA_PAGE_SIZE} -DEIGEN_STACK_ALLOCATION_LIMIT=0 -I ${EIGEN_INCLUDE}  -I include `pkg-config nlohmann_json sndfile --cflags` -I vendored/vectorclass2

ifeq ($(COLOR),1)
CXXFLAGS += -fdiagnostics-color=always
endif

LDFLAGS ?= `pkg-config sndfile --libs` 

examples = ringbuffer nam_wavenet inplace_sequence iterated_sequence conv1d_buffered conv1d2by2_buffered non_nam_wavenet 
benchmarks = mul_raw mul_raw_no_init mul_eigen mul_intrinsics mul_blitz mul_blitz_tiny mul_vectorclass2 conv1d non_nam_wavenet dot process_conv1d mul_row_vs_col
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
	$(CXX) $(CXXFLAGS) `pkg-config benchmark --cflags` -fno-omit-frame-pointer $< $(LDFLAGS) `pkg-config benchmark --libs` -o $@
#	$(CXX) -S $(CXXFLAGS) `pkg-config benchmark --cflags` -fno-omit-frame-pointer $< $(LDFLAGS) `pkg-config benchmark --libs` -o "$@".s

build/tests/%: tests/%.cpp include/anna/*.hpp
	$(CXX) $(CXXFLAGS) `pkg-config gtest_main --cflags` $< `pkg-config gtest_main --libs` -o $@

clean:
	rm -f build/examples/*
