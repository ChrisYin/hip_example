#include "hip/hip_runtime.h"

#define WIDTH 8
#define HEIGHT 8

extern "C"  __global__ void vectoradd_int(int *__restrict__ a, const int *__restrict__ b, const int *__restrict__ c)

{
	int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
	int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

	int i = y * WIDTH + x;
	if (i < (WIDTH * HEIGHT)) {
		a[i] = b[i] + c[i];
	}

	asm volatile("s_sethalt 0x1");
}
