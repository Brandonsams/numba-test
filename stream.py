from numba import cuda
import numpy as np

cuda_stream = cuda.stream()
for i in range(1000):
    ary = np.arange(i,i+10)
    d_ary = cuda.to_device(ary, stream=cuda_stream)
    h_ary = d_ary.copy_to_host(stream=cuda_stream)
    print(h_ary)