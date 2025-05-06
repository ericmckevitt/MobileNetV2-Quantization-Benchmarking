

| Device     | Precision        | GPU Compute Time Mean (ms) |
|------------|-------------------|----------------------------|
| Orin NX    | Full-Precision    | 0.940353                   |
| Orin NX    | Quantized         | 0.838389                   |
| Xavier AGX | Full-Precision    | 1.41687                    |
| Xavier AGX | Quantized         | 0.936569                   |

| Device     | Precision         | GPU Compute Time Mean (ms) | % Change vs Full-Precision |
|------------|-------------------|----------------------------|-----------------------------|
| Orin NX    | Full-Precision    | 0.940353                   |                             |
| Orin NX    | Quantized         | 0.838389                   | **−10.8%**                  |
| Xavier AGX | Full-Precision    | 1.41687                    |                             |
| Xavier AGX | Quantized         | 0.936569                   | **−33.9%**                  |

---

# Orin NX CUDA API Time Breakdown

| Operation               | FP32 (ms / % total) | INT8 (ms / % total) |
|-------------------------|---------------------|---------------------|
| cudaEventSynchronize    | 26,885.56 / 30.3%   | 26,530.35 / 29.4%   |
| cudaStreamSynchronize   | 13,055.48 / 14.7%   | 16,614.04 / 18.4%   |
| cuLaunchKernelEx        | 12,622.56 / 14.2%   | 13,401.57 / 14.9%   |
| cudaMalloc              |  9,439.76 / 10.6%   |  7,111.87 / 7.9%    |
| cudaFree / cudaMemset   |  7,320.76 / 8.3%    |  6,988.81 / 7.8%    |

# Xavier AGX CUDA API Time Breakdown


| Operation              | FP32 (ms / % total) | INT8 (ms / % total) |
|------------------------|---------------------|---------------------|
| cudaLaunchKernel       | 64748.82 / 40.93%   | 35920.19 / 26.05%   |
| cudaEventSynchronize   | 61391.92 / 38.81%   | 72214.36 / 52.37%   |
| cudaStreamWaitEvent    |  5959.55 /  3.77%   | —                   |
| cudaEventRecord        |  5741.43 /  3.63%   |  4208.84 /  3.05%   |
| cudaStreamAddCallback  |  4683.14 /  2.96%   |  6100.38 /  4.42%   |
| cudaMemcpyAsync        | —                   |  4865.33 /  3.53%   |
