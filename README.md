# gpuAllocator

pre-allocate some CUDA VRAM and hold on to it, til ctrl+c is pressed, pass in MB on command line or 1024MB will be the default allocation.

Linux and Windows should work ( SLN for VS 2022 supplied, Linux Makefile added tested under WSL2/Ubuntu)


# Usage

  gpuAllocator <size_in_mb> [device_index]

size_in_mb numerical size in MB to allocate, example 1024 
optional device_index is cuda device number in a multi gpu syste,
