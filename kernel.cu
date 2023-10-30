#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <csignal>
#include <cstdlib>

cudaError_t runGpuMem(unsigned long long memSize);

__global__ void emptyKernel() {
	// do nothing
}


/// <summary>
/// allocates a block of CUDA VRAM and holes on to it, till ctrl +c is pressed. or if 0 is passed display current allocation
/// </summary>
/// <param name="argc">number of arguments</param>
/// <param name="argv">size to allocate in MB or 0 for stats</param>
/// <returns>0 on pass, 1 on fail</returns>
int main(int argc, char* argv[]) {
	unsigned long long memSize = 0;
	cudaError_t cudaStatus;

	// get the amount of memory to allocate in MB, default to 256
	if (argc < 2 || sscanf(argv[1], "%llu", &memSize) != 1) {
		
		std::cout << "No size in MB passed in, defaulting to 1024MB\n\n";

		memSize = 1024;

	}

	memSize *= (1024ULL * 1024ULL);  // convert MB to bytes

	// if memSize is 0, just show available VRAM and exit
	if (memSize == 0) {
		size_t freeMem, totalMem;
		cudaMemGetInfo(&freeMem, &totalMem);
		std::cout << "Available CUDA VRAM: " << freeMem << " bytes free, " << totalMem << " bytes total.\n";
		return 0;
	}

	// allocate memory and keep an active kernel
	cudaStatus = runGpuMem(memSize);
	if (cudaStatus != cudaSuccess) {
		std::cerr << "RunGpuMem failed!\n";
		return 1;
	}

	return 0;
}

/// <summary>
/// allocate a block of VRAM in an empty kernel and wait for CTRL+C
/// </summary>
/// <param name="memSize">amount of memory in MB to allocate</param>
/// <returns>cudaStatus</returns>
cudaError_t runGpuMem(unsigned long long memSize) {
	size_t freeMem, totalMem;
	cudaMemGetInfo(&freeMem, &totalMem);
	std::cout << "Before allocation: " << freeMem << " bytes free, " << totalMem << " bytes total.\n";

	void* gpuMem = nullptr;
	cudaError_t cudaStatus = cudaMalloc(&gpuMem, memSize);

	if (cudaStatus != cudaSuccess) {
		std::cerr << "Error, could not allocate " << memSize << " bytes.\n";
		
	}
	else {
		std::cout << "Allocated " << memSize << " bytes, (" << memSize / (1024ULL * 1024ULL) << " MB).\n";


		cudaMemGetInfo(&freeMem, &totalMem);
		std::cout << "After allocation: " << freeMem << " bytes free, " << totalMem << " bytes total.\n";

		// inform user how to exit
		std::cout << "Press CTRL+C to exit...\n";

		while (true) {

			// launch a kernel on the GPU with one thread
			emptyKernel <<<1, 1 >>> ();

			// check for any errors launching the kernel
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				std::cerr << "Kernel launch failed: " << cudaGetErrorString(cudaStatus) << "\n";
				break;
			}

			// cudaDeviceSynchronize waits for the kernel to finish
			cudaStatus = cudaDeviceSynchronize();
			if (cudaStatus != cudaSuccess) {
				std::cerr << "CudaDeviceSynchronize returned error code " << cudaStatus << " after launching kernel!\n";
				break;
			}
		}
	}

	// free GPU memory and exit
	cudaFree(gpuMem);

	return cudaStatus;
}
