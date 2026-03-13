# GPU Accelerated Image Edge Detection using PyCUDA

## Project Description

This project implements **Sobel edge detection** on images using GPU-accelerated CUDA kernels with PyCUDA. It applies custom CUDA kernels for parallel edge detection and compares CPU vs GPU performance. Each GPU thread processes one pixel, demonstrating the power of GPU parallelism for image processing tasks.

## Dataset Explanation

This project uses images from the **USC SIPI Image Database (Miscellaneous category)** - high-quality test images commonly used in image processing research:
These diverse images provide varied edge characteristics for robust edge detection testing.

## Quickstart

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the pipeline:
```bash
bash run.sh
```

Or run directly:
```bash
python3 src/edge_detection.py
```

## How to Run

### Full Pipeline
```bash
bash run.sh
```

### Direct Python Command
```bash
python3 src/edge_detection.py
```

### To Process Single Image
```python
from edge_detection import EdgeDetector
detector = EdgeDetector()
cpu_time, gpu_time = detector.process_image('misc/boat.512.tiff')
print(f"Speedup: {cpu_time/gpu_time:.2f}x")
```

## CLI Commands

**Run full pipeline:**
```bash
bash run.sh
python3 src/edge_detection.py
```

**Check GPU availability:**
```bash
python3 -c "import pycuda; print('GPU available')"
nvidia-smi
```

**View results:**
```bash
cat results/execution_times.csv
python3 -c "import pandas as pd; print(pd.read_csv('results/execution_times.csv'))"
```

**List output images:**
```bash
ls -lh output/
find output -name "*.png" | wc -l
```

**Process with custom paths:**
```bash
python3 << 'EOF'
from edge_detection import EdgeDetector
detector = EdgeDetector()
detector.process_batch(image_dir='misc', output_dir='output', results_dir='results')
EOF
```

**Build with CMake:**
```bash
mkdir build && cd build
cmake -DCUDA_ARCH=75 ..
make
```

## Code Organization

- **src/edge_detection.py** - Main Python implementation with CPU and GPU edge detection
- **src/sobel_kernel.cu** - CUDA kernels for GPU edge detection (basic + optimized versions)
- **run.sh** - Shell script to run the pipeline
- **CMakeLists.txt** - CMake build configuration
- **misc/** - Input images from USC SIPI database (41 images)
- **output/** - Edge-detected output images (auto-created)
- **results/** - Performance metrics CSV (auto-created)

## Expected Outputs

**Edge Images** (in `output/`):
- `{image_name}_edges_cpu.png` - CPU-based edge detection
- `{image_name}_edges_gpu.png` - GPU-based edge detection

**Performance Metrics** (in `results/execution_times.csv`):
```
image,cpu_time_ms,gpu_time_ms,speedup
boat.512.tiff,245.32,18.45,13.30
house.tiff,156.78,12.34,12.71
```

## Sobel Edge Detection

The Sobel operator uses two 3×3 kernels:

**Gx (horizontal gradient):**
```
[-1  0  +1]
[-2  0  +2]
[-1  0  +1]
```

**Gy (vertical gradient):**
```
[-1  -2  -1]
[ 0   0   0]
[+1  +2  +1]
```

**Edge Magnitude:** `G = sqrt(Gx² + Gy²)`

## GPU Acceleration

- **Parallelism**: Each thread processes one pixel
- **Block Size**: 32×8 threads per block (optimized)
- **Memory**: Uses CUDA shared memory for efficient caching
- **Speedup**: 10-50x faster than CPU (depends on GPU)
- **Kernels**: Both basic and shared-memory optimized versions included

## Installation

**Requirements:**
- Python 3.8+
- NVIDIA GPU with CUDA Compute Capability 3.0+
- CUDA Toolkit 11.0+

**Install:**
```bash
pip install -r requirements.txt
```

## Dependencies

- PyCUDA - GPU programming
- NumPy - Numerical computing
- OpenCV - Image I/O
- SciPy - Scientific computing
- Pandas - CSV handling

## References

- Sobel Operator: https://en.wikipedia.org/wiki/Sobel_operator
- PyCUDA: https://documen.tician.de/pycuda/
- USC SIPI: http://sipi.usc.edu/database/
- CUDA Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/

---
