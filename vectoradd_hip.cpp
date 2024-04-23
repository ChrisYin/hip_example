/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include <iostream>
#include "hip/hip_runtime.h"

#define HIP_ASSERT(x) (assert((x) == hipSuccess))

#define WIDTH 8
#define HEIGHT 8

#define NUM (WIDTH * HEIGHT)

#define THREADS_PER_BLOCK_X 2
#define THREADS_PER_BLOCK_Y 4
#define THREADS_PER_BLOCK_Z 1

#define fileName "vectoradd_kernel.code"
#define kernel_name "vectoradd_int"

using namespace std;

int main()
{
	int *hostA;
	int *hostB;
	int *hostC;

	int *deviceA;
	int *deviceB;
	int *deviceC;

	hipDeviceProp_t devProp;
	hipGetDeviceProperties(&devProp, 0);
	cout << " System minor " << devProp.minor << endl;
	cout << " System major " << devProp.major << endl;
	cout << " agent prop name " << devProp.name << endl;

	cout << "hip Device prop succeeded " << endl;

	int i, j;
	int errors;

	hostA = (int *)malloc(NUM * sizeof(int));
	hostB = (int *)malloc(NUM * sizeof(int));
	hostC = (int *)malloc(NUM * sizeof(int));

	// initialize the input data
	for (i = 0; i < NUM; i++) {
		hostB[i] = (int)i;
		hostC[i] = (int)100;
	}

	HIP_ASSERT(hipMalloc((void **)&deviceA, NUM * sizeof(int)));
	HIP_ASSERT(hipMalloc((void **)&deviceB, NUM * sizeof(int)));
	HIP_ASSERT(hipMalloc((void **)&deviceC, NUM * sizeof(int)));

	HIP_ASSERT(hipMemcpy(deviceB, hostB, NUM * sizeof(int), hipMemcpyHostToDevice));
	HIP_ASSERT(hipMemcpy(deviceC, hostC, NUM * sizeof(int), hipMemcpyHostToDevice));

	hipModule_t Module;
	hipFunction_t Function;
	hipModuleLoad(&Module, fileName);
	hipModuleGetFunction(&Function, Module, kernel_name);

	void *args[3] = { &deviceA, &deviceB, &deviceC };

	hipModuleLaunchKernel(Function, WIDTH / THREADS_PER_BLOCK_X, HEIGHT / THREADS_PER_BLOCK_Y, 1,
			      THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, 1, 0, 0, args, nullptr);

	HIP_ASSERT(hipMemcpy(hostA, deviceA, NUM * sizeof(int), hipMemcpyDeviceToHost));

	// verify the results
	errors = 0;
	for (i = 0; i < NUM; i++) {
		if (hostA[i] != (hostB[i] + hostC[i])) {
			errors++;
		}
	}
	if (errors != 0) {
		printf("FAILED: %d errors\n", errors);
	} else {
		printf("PASSED!\n");
	}
	for (i = 0; i < HEIGHT; i++) {
		for (j = 0; j < WIDTH; j++) {
			int idx = i * WIDTH + j;
			int grid_idx_x = i / THREADS_PER_BLOCK_X;
			int grid_idx_y = j / THREADS_PER_BLOCK_Y;
			printf("%d ", hostA[idx]);
		}
		printf("\n");
	}

	HIP_ASSERT(hipFree(deviceA));
	HIP_ASSERT(hipFree(deviceB));
	HIP_ASSERT(hipFree(deviceC));

	free(hostA);
	free(hostB);
	free(hostC);

	//hipResetDefaultAccelerator();

	return errors;
}
