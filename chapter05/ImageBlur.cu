/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.
 * Any use, reproduction, disclosure, or distribution of this software
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA)
 * associated with this source code for terms and conditions that govern
 * your use of this NVIDIA software.
 *
 */

#include "cuda.h"
#include "../common/book.h"
#include "../common/cpu_bitmap.h"

#define DIM 1024

__global__ void kernel( unsigned char *ptr ) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    __shared__ unsigned char    shared[16][16];

    // now calculate the value at that position
    shared[threadIdx.x][threadIdx.y] = ptr[offset]; //MIGHT WORK?

    // removing this syncthreads shows graphically what happens
    // when it doesn't exist.  this is an example of why we need it.
    __syncthreads();
 		if (x == 0) x++;
		if (x == DIM-1) x--;
		if (y == 0) y++;
		if (y == DIM-1) y--;

		unsigned char t, l, c, r, b, tl, tr, bl, br, average;
		t = shared[threadIdx.x][threadIdx.y+1];
		l = shared[threadIdx.x-1][threadIdx.y];
		c = shared[threadIdx.x][threadIdx.y];
		r = shared[threadIdx.x+1][threadIdx.y];
		b = shared[threadIdx.x][threadIdx.y-1];
		tl = shared[threadIdx.x-1][threadIdx.y+1];
		tr = shared[threadIdx.x+1][threadIdx.y+1];
		bl = shared[threadIdx.x-1][threadIdx.y-1];
		br = shared[threadIdx.x+1][threadIdx.y-1];
		average = (t+l+c+r+b+tl+tr+bl+br)/9;
    ptr[offset = average;
}

// globals needed by the update routine
struct DataBlock {
    unsigned char   *dev_bitmap;
};

struct cuComplex {
    float   r;
    float   i;
    cuComplex( float a, float b ) : r(a), i(b)  {}
    float magnitude2( void ) { return r * r + i * i; }
    cuComplex operator*(const cuComplex& a) {
        return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }
    cuComplex operator+(const cuComplex& a) {
        return cuComplex(r+a.r, i+a.i);
    }
};

int julia( int x, int y ) {
    const float scale = 1.5;
    float jx = scale * (float)(DIM/2 - x)/(DIM/2);
    float jy = scale * (float)(DIM/2 - y)/(DIM/2);

    cuComplex c(-0.8, 0.156);
    cuComplex a(jx, jy);

    int i = 0;
    for (i=0; i<200; i++) {
        a = a * a + c;
        if (a.magnitude2() > 1000)
            return 0;
    }

    return 1;
}

void draw( unsigned char *ptr ){
    for (int y=0; y<DIM; y++) {
        for (int x=0; x<DIM; x++) {
            int offset = x + y * DIM;

            int juliaValue = julia( x, y );
            ptr[offset*4 + 0] = 255 * juliaValue;
            ptr[offset*4 + 1] = 0;
            ptr[offset*4 + 2] = 0;
            ptr[offset*4 + 3] = 255;
        }
    }
 }

int main( void ) {
	DataBlock   data;
	CPUBitmap bitmap( DIM, DIM, &data );
	unsigned char    *dev_bitmap;
	unsigned char *ptr = bitmap.get_ptr();
	draw (ptr);

	HANDLE_ERROR( cudaMalloc( (void**)&dev_bitmap,
														bitmap.image_size() ) );
	data.dev_bitmap = dev_bitmap;

	dim3    grids(DIM/16,DIM/16);
	dim3    threads(16,16);
	kernel<<<grids,threads>>>( dev_bitmap );

	HANDLE_ERROR( cudaMemcpy( bitmap.get_ptr(), dev_bitmap,
														bitmap.image_size(),
														cudaMemcpyDeviceToHost ) );

	HANDLE_ERROR( cudaFree( dev_bitmap ) );

	bitmap.display_and_exit();
}
