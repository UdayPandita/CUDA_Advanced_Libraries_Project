#!/usr/bin/env python3

import os
import cv2
import numpy as np
import pandas as pd
import time
from pathlib import Path
import warnings
import sys

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
    PYCUDA_AVAILABLE = True
except ImportError:
    PYCUDA_AVAILABLE = False
    warnings.warn("PyCUDA not available. GPU processing will be skipped.")


class EdgeDetector:
    
    def __init__(self):
        self.cuda_available = PYCUDA_AVAILABLE
        self.kernel = None
        self.block_size = (32, 8)
        
        if self.cuda_available:
            self._compile_cuda_kernel()
    
    def _compile_cuda_kernel(self):
        try:
            kernel_path = Path(__file__).parent / "sobel_kernel.cu"
            
            if not kernel_path.exists():
                cuda_code = self._get_inline_kernel()
            else:
                with open(kernel_path, 'r') as f:
                    cuda_code = f.read()
            
            module = SourceModule(cuda_code)
            
            self.kernel = module.get_function("sobel_edge_detection")
            print("✓ CUDA kernel compiled successfully")
            
        except Exception as e:
            print(f"✗ Failed to compile CUDA kernel: {e}")
            self.cuda_available = False
    
    def _get_inline_kernel(self):
        return """
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
                
                output[y * cols + x] = normalized;
            } else if (x < cols && y < rows) {
                output[y * cols + x] = 0.0f;
            }
        }
        """
    
    def sobel_cpu(self, image):
        sobel_x = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=np.float32)
        
        sobel_y = np.array([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]], dtype=np.float32)
        
        img_float = image.astype(np.float32)
        
        from scipy.ndimage import convolve
        
        gx = convolve(img_float, sobel_x, mode='constant')
        gy = convolve(img_float, sobel_y, mode='constant')
        
        magnitude = np.sqrt(gx**2 + gy**2)
        
        magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)
        
        return magnitude
    
    def sobel_gpu(self, image):
        if not self.cuda_available or self.kernel is None:
            raise RuntimeError("CUDA kernel not available. Falling back to CPU.")
        
        rows, cols = image.shape
        
        input_gpu = cuda.mem_alloc(image.nbytes)
        cuda.memcpy_htod(input_gpu, np.ascontiguousarray(image))
        
        output = np.zeros((rows, cols), dtype=np.float32)
        output_gpu = cuda.mem_alloc(output.nbytes)
        
        block_x, block_y = self.block_size
        grid_x = (cols + block_x - 1) // block_x
        grid_y = (rows + block_y - 1) // block_y
        
        self.kernel(
            input_gpu,
            output_gpu,
            np.int32(rows),
            np.int32(cols),
            block=(block_x, block_y, 1),
            grid=(grid_x, grid_y)
        )
        
        cuda.memcpy_dtoh(output, output_gpu)
        
        input_gpu.free()
        output_gpu.free()
        
        output = np.clip(output, 0, 255).astype(np.uint8)
        
        return output
    
    def process_image(self, image_path, save_output=True, output_dir="output"):
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        
        image_name = Path(image_path).stem
        
        start_cpu = time.time()
        edges_cpu = self.sobel_cpu(image)
        cpu_time = (time.time() - start_cpu) * 1000
        
        gpu_time = None
        edges_gpu = None
        
        if self.cuda_available:
            try:
                start_gpu = time.time()
                edges_gpu = self.sobel_gpu(image)
                gpu_time = (time.time() - start_gpu) * 1000
            except Exception as e:
                print(f"   ⚠ GPU processing failed: {e}")
                gpu_time = None
        
        if save_output:
            os.makedirs(output_dir, exist_ok=True)
            
            if edges_gpu is not None:
                output_path = Path(output_dir) / f"{image_name}_edges_gpu.png"
                cv2.imwrite(str(output_path), edges_gpu)
            
            output_path = Path(output_dir) / f"{image_name}_edges_cpu.png"
            cv2.imwrite(str(output_path), edges_cpu)
        
        return cpu_time, gpu_time
    
    def process_batch(self, image_dir="misc", output_dir="output", results_dir="results"):
        results = []
        image_path = Path(image_dir)
        
        image_files = list(image_path.glob("*.tiff")) + \
                     list(image_path.glob("*.png")) + \
                     list(image_path.glob("*.jpg"))
        
        if not image_files:
            print(f"✗ No images found in {image_dir}")
            return pd.DataFrame()
        
        print(f"✓ Found {len(image_files)} images to process\n")
        
        for idx, img_path in enumerate(image_files, 1):
            try:
                print(f"[{idx}/{len(image_files)}] Processing: {img_path.name}")
                
                cpu_time, gpu_time = self.process_image(
                    img_path,
                    save_output=True,
                    output_dir=output_dir
                )
                
                speedup = cpu_time / gpu_time if gpu_time else None
                
                print(f"   CPU Time:  {cpu_time:.2f} ms")
                if gpu_time:
                    print(f"   GPU Time:  {gpu_time:.2f} ms")
                    print(f"   Speedup:   {speedup:.2f}x")
                print()
                
                results.append({
                    'image': img_path.name,
                    'cpu_time_ms': round(cpu_time, 2),
                    'gpu_time_ms': round(gpu_time, 2) if gpu_time else None,
                    'speedup': round(speedup, 2) if speedup else None
                })
                
            except Exception as e:
                print(f"   ✗ Error processing {img_path.name}: {e}\n")
                continue
        
        df = pd.DataFrame(results)
        
        os.makedirs(results_dir, exist_ok=True)
        results_path = Path(results_dir) / "execution_times.csv"
        df.to_csv(results_path, index=False)
        
        print(f"\n✓ Results saved to {results_path}")
        print(f"\nPerformance Summary:")
        print(df.to_string(index=False))
        
        return df


def main():
    print("=" * 70)
    print("GPU Accelerated Image Edge Detection using PyCUDA")
    print("=" * 70)
    print()
    
    if PYCUDA_AVAILABLE:
        print("✓ PyCUDA detected - GPU acceleration enabled")
    else:
        print("⚠ PyCUDA not available - CPU only mode")
    
    print()
    
    detector = EdgeDetector()
    
    print()
    
    results_df = detector.process_batch(
        image_dir="misc",
        output_dir="output",
        results_dir="results"
    )
    
    print()
    print("=" * 70)
    print("Processing Complete!")
    print("=" * 70)
    
    if not results_df.empty and 'speedup' in results_df.columns:
        speedups = results_df['speedup'].dropna()
        if len(speedups) > 0:
            print(f"\nAverage Speedup: {speedups.mean():.2f}x")
            print(f"Max Speedup:     {speedups.max():.2f}x")
            print(f"Min Speedup:     {speedups.min():.2f}x")


if __name__ == "__main__":
    main()
