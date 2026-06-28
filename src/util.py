import torch

# vectorised version of old fetch_mask_indices
def fetch_mask_indices(mask: torch.Tensor) -> tuple:
    return tuple(mask.nonzero(as_tuple=False).squeeze(1).tolist())

# vectorised version of old generate_permutation_matrix
def generate_permutation_matrix(input_state_space: list) -> torch.Tensor:
    result = torch.cartesian_prod(*input_state_space)
    if result.dim() == 1:
        result = result.unsqueeze(1)
    return result
