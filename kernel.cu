#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include "common/book.h"
#include "common/cpu_bitmap.h"

#define INF     2e10f
#define DIM     1024
#define USE_CONSTANT 1
#define rnd(x) (x * rand() / RAND_MAX)
#define SPHERES 20

struct Sphere
{
    float r, b, g;
    float radius;
    float x, y, z;
    __device__ float hit(float ox, float oy, float* n)
    {
        float dx = ox - x;
        float dy = oy - y;
        if (dx * dx + dy * dy < radius * radius)
        {
            float dz = sqrtf(radius * radius - dx * dx - dy * dy);
            *n = dz / sqrtf(radius * radius);
            return dz + z;
        }
        return -INF;
    }
};

// ====== 구 배열 저장 위치 선택 ======
#if USE_CONSTANT
__constant__ Sphere d_spheres_c[SPHERES];  // 상수 메모리
#else
Sphere* d_spheres_g = nullptr;             // 글로벌 메모리
#endif

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

Sphere* s;
__global__ void kernel(unsigned char* ptr)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * DIM; //
    float ox = (x - DIM / 2);
    float oy = (y - DIM / 2);
    float r = 0, g = 0, b = 0;
    float maxz = -INF;
    // 모든 구와 교차 검사 → 가장 카메라에 가까운 구 선택
    for (int i = 0; i < SPHERES; ++i)
    {
        float n;
        float t = d_spheres_c[i].hit(ox, oy, &n);
        if (t > maxz)
        {
            maxz = t; // 
            float fscale = n;
            r = d_spheres_c[i].r * fscale;
            g = d_spheres_c[i].g * fscale;
            b = d_spheres_c[i].b * fscale;
        }
    }

    ptr[offset * 4 + 0] = (int)(r * 255);
    ptr[offset * 4 + 1] = (int)(g * 255);
    ptr[offset * 4 + 2] = (int)(b * 255);
    ptr[offset * 4 + 3] = 255;
}
int main(void)
{
    // 1) 이벤트로 타이밍 준비
    cudaEvent_t start, stop;
    HANDLE_ERROR( cudaEventCreate(&start) );
    HANDLE_ERROR( cudaEventCreate(&stop) );
    HANDLE_ERROR( cudaEventRecord(start, 0) );

    // 2) 출력 비트맵 준비
    CPUBitmap bitmap(DIM, DIM);
    unsigned char* dev_bitmap;

    HANDLE_ERROR(cudaMalloc((void**)& dev_bitmap, bitmap.image_size()));
    HANDLE_ERROR(cudaMalloc((void**)&s, sizeof(Sphere) * SPHERES)); // 구 배열 (input 배열)

    // 3) 호스트에서 구들 난수로 생성
    Sphere* temp_s = (Sphere*)malloc(sizeof(Sphere) * SPHERES);
    for (int i = 0; i < SPHERES; ++i) {
        temp_s[i].r = rnd(1.0f);
        temp_s[i].g = rnd(1.0f);
        temp_s[i].b = rnd(1.0f);
        temp_s[i].x = rnd(1000.0f) - 500.f;
        temp_s[i].y = rnd(1000.0f) - 500.f;
        temp_s[i].z = rnd(1000.0f) - 500.f;
        temp_s[i].radius = rnd(100.0f) + 20.f;
    }

    // 4) 구 데이터를 GPU로 복사 (버전에 따라 다름) -> kernel실행시킬 준비 완료

#if USE_CONSTANT
    HANDLE_ERROR(cudaMemcpyToSymbol(d_spheres_c, temp_s, sizeof(Sphere) * SPHERES));
#else
    HANDLE_ERROR(cudaMalloc((void**)&d_spheres_g, sizeof(Sphere) * SPHERES));
    HANDLE_ERROR(cudaMemcpy(d_spheres_g, temp_s, sizeof(Sphere) * SPHERES, cudaMemcpyHostToDevice));
#endif

    HANDLE_ERROR(cudaMemcpy(s, temp_s, sizeof(Sphere) * SPHERES, cudaMemcpyHostToDevice));
    free(temp_s);

    // 5) 커널 실행을 위해 bitmap generate
	dim3 grids(DIM / 16, DIM / 16);
	dim3 threads(16, 16);
    kernel<<<grids, threads >>> (dev_bitmap);

    HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));
    bitmap.display_and_exit();

    cudaFree(dev_bitmap);
    cudaFree(s);
}
/*
int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
*/
// Helper function for using CUDA to add vectors in parallel.
/*
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
*/