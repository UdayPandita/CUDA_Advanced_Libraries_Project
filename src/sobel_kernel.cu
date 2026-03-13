#include <math.h>

__global__ void sobel_edge_detection(unsigned char *input, float *output, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x > 0 && x < cols - 1 && y > 0 && y < rows - 1) {
        unsigned char p0 = input[(y - 1) * cols + (x - 1)];
        unsigned char p1 = input[(y - 1) * cols + x];
        unsigned char p2 = input[(y - 1) * cols + (x + 1)];
        unsigned char p3 = input[y * cols + (x - 1)];
        unsigned char p5 = input[y * cols + (x + 1)];
        unsigned char p6 = input[(y + 1) * cols + (x - 1)];
        unsigned char p7 = input[(y + 1) * cols + x];
        unsigned char p8 = input[(y + 1) * cols + (x + 1)];
        
        float Gx = -p0 + p2 - 2*p3 + 2*p5 - p6 + p8;
        float Gy = -p0 - 2*p1 - p2 + p6 + 2*p7 + p8;
        
        float magnitude = sqrtf(Gx * Gx + Gy * Gy);
        float normalized = (magnitude / 1024.0f) * 255.0f;
        
        if (normalized > 255.0f) normalized = 255.0f;
        if (normalized < 0.0f) normalized = 0.0f;
        
        output[y * cols + x] = normalized;
    } else if (x < cols && y < rows) {
        output[y * cols + x] = 0.0f;
    }
}

__global__ void sobel_edge_detection_optimized(unsigned char *input, float *output, int rows, int cols) {
    __shared__ unsigned char shared[(32+2)][(32+2)];
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    if (x < cols && y < rows) {
        shared[ty + 1][tx + 1] = input[y * cols + x];
        
        if (tx == 0 && x > 0) {
            shared[ty + 1][tx] = input[y * cols + (x - 1)];
        }
        if (tx == blockDim.x - 1 && x < cols - 1) {
            shared[ty + 1][tx + 2] = input[y * cols + (x + 1)];
        }
        if (ty == 0 && y > 0) {
            shared[ty][tx + 1] = input[(y - 1) * cols + x];
        }
        if (ty == blockDim.y - 1 && y < rows - 1) {
            shared[ty + 2][tx + 1] = input[(y + 1) * cols + x];
        }
    }
    
    __syncthreads();
    
    if (x > 0 && x < cols - 1 && y > 0 && y < rows - 1) {
        unsigned char p0 = shared[ty][tx];
        unsigned char p1 = shared[ty][tx + 1];
        unsigned char p2 = shared[ty][tx + 2];
        unsigned char p3 = shared[ty + 1][tx];
        unsigned char p5 = shared[ty + 1][tx + 2];
        unsigned char p6 = shared[ty + 2][tx];
        unsigned char p7 = shared[ty + 2][tx + 1];
        unsigned char p8 = shared[ty + 2][tx + 2];
        
        float Gx = -p0 + p2 - 2*p3 + 2*p5 - p6 + p8;
        float Gy = -p0 - 2*p1 - p2 + p6 + 2*p7 + p8;
        
        float magnitude = sqrtf(Gx * Gx + Gy * Gy);
        float normalized = (magnitude / 1024.0f) * 255.0f;
        
        if (normalized > 255.0f) normalized = 255.0f;
        
        output[y * cols + x] = normalized;
    } else if (x < cols && y < rows) {
        output[y * cols + x] = 0.0f;
    }
}
