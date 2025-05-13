import numpy as np
import scipy
import torch
import rasterio
import matplotlib.pyplot as plt

print(f"NumPy version: {np.__version__}")
print(f"SciPy version: {scipy.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"Rasterio version: {rasterio.__version__}")

# Test basic NumPy operations
test_array = np.array([[1, 2, 3], [4, 5, 6]])
print(f"\nNumPy array operations:")
print(f"Original array:\n{test_array}")
print(f"Array shape: {test_array.shape}")
print(f"Array sum: {test_array.sum()}")
print(f"Array mean: {test_array.mean()}")

# Test basic PyTorch operations
test_tensor = torch.tensor(test_array)
print(f"\nPyTorch tensor operations:")
print(f"Original tensor:\n{test_tensor}")
print(f"Tensor shape: {test_tensor.shape}")
print(f"Tensor sum: {test_tensor.sum().item()}")
print(f"Tensor mean: {test_tensor.float().mean().item()}")

print("\nEnvironment test completed successfully!")