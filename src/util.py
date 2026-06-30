import torch

# Vectorised version of old fetch_mask_indices
# Return the indices associated with a '1' value as a tuple
def fetch_mask_indices(mask: torch.Tensor) -> tuple:
    indices = mask.nonzero(as_tuple=False).squeeze(1)
    return tuple(indices.tolist())

# Vectorised version of old generate_permutation_matrix
# Return a matrix which contains all input permutations
# input_state_space is a N dimensional list of all input tensors as their state space 
# fully listed in said tensor
def generate_permutation_matrix(input_state_space: list) -> torch.Tensor:
    permutations = torch.cartesian_prod(*input_state_space)
    if(permutations.dim() == 1):
        permutations = permutations.unsqueeze(1)
    return permutations
    
