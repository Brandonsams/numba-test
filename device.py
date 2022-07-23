from numba import cuda
cuda.select_device(0)
# dev = cuda.current_context().device
# prints e.g. "GPU-e6489c45-5b68-3b03-bab7-0e7c8e809643"
for gpu in cuda.gpus:
    print(gpu._device.name)

print(cuda.detect())