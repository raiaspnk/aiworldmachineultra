import torch
import torch.nn.functional as F

kernel_sz = 31
pad = kernel_sz // 2

depth_tensor = torch.rand((1024, 1024)).cuda()
print("Original Contiguous: ", depth_tensor.is_contiguous())

depth_unsqueezed = depth_tensor.unsqueeze(0).unsqueeze(0)
eroded = -F.max_pool2d(-depth_unsqueezed, kernel_sz, stride=1, padding=pad)
terrain_depth_tensor = F.max_pool2d(eroded, kernel_sz, stride=1, padding=pad).squeeze()

print("Eroded Contiguous: ", eroded.is_contiguous())
print("Terrain Contiguous: ", terrain_depth_tensor.is_contiguous())

