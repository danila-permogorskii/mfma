#include <hip/hip_runtime.h>

#include <cstdlib>
#include <iostream>
#include <vector>

int main() {
	hipDeviceProp_t props;
	hipGetDeviceProperties(&props, 0 /*deviceID*/);
	std::cout << "Info: running on device" << props.name << "\n";
}
