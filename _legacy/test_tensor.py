import numpy as np

# Create a sample array instead of tensor
depth = np.array([[100.0, 50.0], [20.0, 5.0]], dtype=np.float32)

# Save
np.savez_compressed('test_tensor.npz', depth=depth)

# Load it back
data = np.load('test_tensor.npz', allow_pickle=True)
loaded_depth = data['depth']

print("Type: ", type(loaded_depth))
print("Shape: ", loaded_depth.shape)
print("Dtype: ", loaded_depth.dtype)
print("Content: ", loaded_depth)

# In world_generator.py, we do:
min_d, max_d = loaded_depth.min(), loaded_depth.max()
print(f"Min: {min_d}, Max: {max_d}")

try:
    inverted = (max_d - loaded_depth) / (max_d - min_d)
    print("Inverted successfully:")
    print(inverted)
except Exception as e:
    print(f"Failed to invert: {e}")
    
try:
    casted = inverted.astype(np.float32)
    print("Casted successfully.")
except Exception as e:
    print(f"Failed to cast: {e}")
