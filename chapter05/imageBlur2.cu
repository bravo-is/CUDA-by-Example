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
#include "../common/cpu_anim.h"

#define DIM 1024

// these exist on the GPU side
texture<float,2>  texIn;
texture<float,2>  texOut;

__global__ void GOL_kernel( float *dst, bool dstOut ) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float   t, l, c, r, b, tl, tr, bl, br, average;
    if (dstOut) {
      t = tex2D(texIn,x,y-1);//top
      l = tex2D(texIn,x-1,y);//left
      c = tex2D(texIn,x,y);//center
      r = tex2D(texIn,x+1,y);//right
      b = tex2D(texIn,x,y+1);//bottom
      tl = tex2D(texIn,x-1,y-1);//top-left
      tr = tex2D(texIn,x+1,y-1);//top-right
      bl = tex2D(texIn,x-1,y+1);//bottom-left
      br = tex2D(texIn,x+1,y+1);//bottom-right
    }else{
      t = tex2D(texOut,x,y-1);//top
      l = tex2D(texOut,x-1,y);//left
      c = tex2D(texOut,x,y);//center
      r = tex2D(texOut,x+1,y);//right
      b = tex2D(texOut,x,y+1);//bottom
      tl = tex2D(texOut,x-1,y-1);//top-left
      tr = tex2D(texOut,x+1,y-1);//top-right
      bl = tex2D(texOut,x-1,y+1);//bottom-left
      br = tex2D(texOut,x+1,y+1);//bottom-right
    }
    average = (t+l+r+b+tl+tr+bl+br+c)/9;
    //boxblur
    dst[offset] = average;
}

// globals needed by the update routine
struct DataBlock {
    unsigned char   *output_bitmap;
    float           *dev_inSrc;
    float           *dev_outSrc;
    CPUAnimBitmap  *bitmap;

    cudaEvent_t     start, stop;
    float           totalTime;
    float           frames;
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

void draw( float *ptr ){
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

void anim_gpu( DataBlock *d, int ticks ) {
    HANDLE_ERROR( cudaEventRecord( d->start, 0 ) );
    dim3    blocks(DIM/16,DIM/16);
    dim3    threads(16,16);
    CPUAnimBitmap  *bitmap = d->bitmap;

    // since tex is global and bound, we have to use a flag to
    // select which is in/out per iteration
    // we maintain this so that cylce speed can be controlled by timesteps or FPS
    volatile bool dstOut = true;
    for (int i=0; i<2; i++) {
        float *out;
        if (dstOut) {
            out = d->dev_outSrc;
        } else {
            out = d->dev_inSrc;
        }
        GOL_kernel<<<blocks,threads>>>( out, dstOut );
        dstOut = !dstOut;
    }
    float_to_color<<<blocks,threads>>>( d->output_bitmap,
                                        d->dev_inSrc );

    HANDLE_ERROR( cudaMemcpy( bitmap->get_ptr(),
                              d->output_bitmap,
                              bitmap->image_size(),
                              cudaMemcpyDeviceToHost ) );

    HANDLE_ERROR( cudaEventRecord( d->stop, 0 ) );
    HANDLE_ERROR( cudaEventSynchronize( d->stop ) );
    float   elapsedTime;
    HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime,
                                        d->start, d->stop ) );
    d->totalTime += elapsedTime;
    ++d->frames;
    printf( "Average Time per frame:  %3.1f ms\n",
            d->totalTime/d->frames  );
}

// clean up memory allocated on the GPU
void anim_exit( DataBlock *d ) {
    cudaUnbindTexture( texIn );
    cudaUnbindTexture( texOut );

    HANDLE_ERROR( cudaFree( d->dev_inSrc ) );
    HANDLE_ERROR( cudaFree( d->dev_outSrc ) );

    HANDLE_ERROR( cudaEventDestroy( d->start ) );
    HANDLE_ERROR( cudaEventDestroy( d->stop ) );
}


int main( void ) {
    DataBlock   data;
    CPUAnimBitmap bitmap( DIM, DIM, &data );
    data.bitmap = &bitmap;
    data.totalTime = 0;
    data.frames = 0;
    HANDLE_ERROR( cudaEventCreate( &data.start ) );
    HANDLE_ERROR( cudaEventCreate( &data.stop ) );

    int imageSize = bitmap.image_size();

    HANDLE_ERROR( cudaMalloc( (void**)&data.output_bitmap,
                               imageSize ) );

    // assume float == 4 chars in size (ie rgba)
    HANDLE_ERROR( cudaMalloc( (void**)&data.dev_inSrc,
                              imageSize ) );
    HANDLE_ERROR( cudaMalloc( (void**)&data.dev_outSrc,
                              imageSize ) );

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

    HANDLE_ERROR( cudaBindTexture2D( NULL, texIn,
                                   data.dev_inSrc,
                                   desc, DIM, DIM,
                                   sizeof(float) * DIM ) );

    HANDLE_ERROR( cudaBindTexture2D( NULL, texOut,
                                   data.dev_outSrc,
                                   desc, DIM, DIM,
                                   sizeof(float) * DIM ) );

    float *inputGrid = (float*)malloc( imageSize );
    for (int i=0; i<DIM*DIM; i++) {
        unputGrid[i] = 0.0f;
    }
    draw( inputGrid );

    HANDLE_ERROR( cudaMemcpy( data.dev_inSrc, inputGrid,
                              imageSize,
                              cudaMemcpyHostToDevice ) );
    free( inputGrid );

    bitmap.anim_and_exit( (void (*)(void*,int))anim_gpu,
                           (void (*)(void*))anim_exit );
}
