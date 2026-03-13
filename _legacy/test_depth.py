import argparse
import numpy as np
import cv2

def test_logic():
    # Simulate a fake metric depth array from DA3 metric (float meters)
    # E.g. 5x5 array where sky is 1000m away, and buildings are 5-50m away
    # DA3 returns metric distance, so smaller values are closer.
    # Sky is highest value, ground/buildings are smaller.
    img_depth = np.array([
        [100.0, 100.0, 100.0, 100.0, 100.0],
        [100.0,  50.0,  50.0, 100.0, 100.0],
        [100.0,  20.0,  20.0, 100.0, 100.0],
        [100.0,   5.0,   5.0, 100.0, 100.0],
        [  1.0,   1.0,   1.0,   1.0,   1.0]  # Floor right at camera
    ], dtype=np.float32)
    
    print("Original Depth (meters):")
    print(img_depth)

    # Inverter Profundidade:
    min_d, max_d = img_depth.min(), img_depth.max()
    print(f"Min: {min_d}, Max: {max_d}")
    if max_d > min_d:
        img_depth = (max_d - img_depth) / (max_d - min_d)
    else:
        img_depth = np.zeros_like(img_depth)
        
    print("After Inversion (0-1), 1 = Closest:")
    print(img_depth)
    
    # Padronizar
    img_depth = img_depth.astype(np.float32)
    
    # Filtro
    kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    img_depth_hpass = cv2.filter2D(img_depth, -1, kernel_sharpen)
    
    print("After Sharpen Filter:")
    print(img_depth_hpass)
    
    # Normalização adaptativa
    max_val = np.max(img_depth_hpass)
    if max_val == 0: max_val = 1.0
    depth_norm = img_depth_hpass.astype(np.float32) / float(max_val)
    
    print("Final Normalized Depth (passed to CUDA):")
    print(depth_norm)

if __name__ == "__main__":
    test_logic()
