# Compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -O3 -arch=sm_35

# Target executable name
EXE = gpuAllocator 

# Source files
SRC = kernel.cu

all: $(EXE)

$(EXE): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(SRC) -o $(EXE)

clean:
	rm -f $(EXE)
