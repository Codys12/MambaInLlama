import torch

def scale_vectors(v1, v2):
    # Compute the L2 norm (magnitude) of each vector
    norm1 = torch.norm(v1)
    norm2 = torch.norm(v2)
    
    # Find the smaller of the two norms
    min_norm = torch.min(norm1, norm2)
    
    # Compute the scaling factor (1/2 of the smallest vector's scale)
    scaling_factor = 0.5 * min_norm
    
    # Scale both vectors
    v1_scaled = v1 * (scaling_factor / norm1)
    v2_scaled = v2 * (scaling_factor / norm2)
    
    return v1_scaled, v2_scaled

# Example usage
v1 = torch.tensor([1.0, 2.0, 3.0])
v2 = torch.tensor([4.0, 5.0, 6.0])

v1_result, v2_result = scale_vectors(v1, v2)

print("Original v1:", v1)
print("Original v2:", v2)
print("Scaled v1:", v1_result)
print("Scaled v2:", v2_result)
print("Norm of scaled v1:", torch.norm(v1_result))
print("Norm of scaled v2:", torch.norm(v2_result))