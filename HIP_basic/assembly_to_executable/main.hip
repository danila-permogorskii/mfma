#include <hip/hip_runtime.h>

#include "example_utils.hpp"

#include <cstdlib>
#include <iostream>
#include <vector>

/// \brief Device function to square each element
/// in the array 'in' and write to array 'out'.

template<typename T>
__global__ void vector_square_kernel(T* out, T* in, const unsigned long long size){
	// Get the unique global thread id
	const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
	// Each thread hop stride amount of elements to find the next
	// element to square
	const size_t stride = blockDim.x * gridDim.x;

	for(size_t i = offset; i < size; i += stride) {
		out[i] = in[i] * in[i];
	}
}

int main() {
	// Set the problem size
	constexpr size_t size = 1000000;
	constexpr size_t size_in_bytes = size * sizeof(float);

	hipDeviceProp_t props;
	HIP_CHECK(hipGetDeviceProperties(&props, 0 /*deviceID*/));
	std::cout << "Info: running on device" << props.name << "\n";

	std::cout << "Info: allocate host mem (" << 2 * size_in_bytes / 1024.0 / 1024.0 << "MiB) " << "\n";

	// Declare the host side arrays
	std::vector<float> h_in(size);
	std::vector<float> h_out(size);

	// Initialise the host size input
	for(size_t i = 0; i < size; i++) {
		h_in[i] = 1.618f + i;
	}

	// Declare the device side arrays
	float *d_in, *d_out;
	std::cout << "info: allocate device mem (" << 2 * size_in_bytes / 1024.0 / 1024.0 << "MiB)\n";
	// Allocate the device side memory
	HIP_CHECK(hipMalloc(&d_in, size_in_bytes));
	HIP_CHECK(hipMalloc(&d_out, size_in_bytes));

	std::cout << "info: copy Host2Device\n";

	// Copy the input from host to the GPU device
	HIP_CHECK(hipMemcpy(d_in, h_in.data(), size_in_bytes, hipMemcpyHostToDevice));

	// Set the number of blocks per kernel grid
	constexpr uint32_t grid_size = 512;
	// Set the number of threads per kernel block
	constexpr uint32_t threads_per_block = 256;

	std::cout << "info: launch 'vector_square_kernel' kernel\n";
	vector_square_kernel<<<grid_size, threads_per_block, 0, hipStreamDefault>>>(d_out, d_in, size);

	// Check that the kernel invocation was successful
	HIP_CHECK(hipGetLastError());

	std::cout << "info: copy Device2Host\n";
	HIP_CHECK(hipMemcpy(h_out.data(), d_out, size_in_bytes, hipMemcpyDeviceToHost));

	HIP_CHECK(hipFree(d_in));
	HIP_CHECK(hipFree(d_out));

	std::cout << "info: check result\n";

	for(size_t i = 0; i < size; i++) {
		if(h_out[i] != h_in[i] * h_in[i]) {
			std::cerr << "FAILED! h_out [" << i << "] =" << h_out[i]
				<< ", expected: " << h_in[i] * h_in[i] << '\n';
			exit(error_exit_code);
		}
	}
	std::cout << "PASSED!\n";
}
