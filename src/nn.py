import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch import Tensor
from torch.nn import init
from .quant import QuantizerBase
from .util import generate_permutation_matrix

# Walks the tree and generates truth tables for every neuron 
def generate_truth_tables(model: nn.Module, verbose: bool = False) -> None:
    model.eval()
    for name, module in model.named_modules():
        if isinstance(module, LUTLayer):
            if verbose:
                print(f"Calculating truth tables for {name}")
            module.calculate_truth_tables()

class SparseLinear(nn.Linear):
    def __init__(self, 
                 in_features: int,
                 out_features: int,
                 bias: bool = True):
        super().__init__(in_features, out_features, bias)

        # initialise the weight matrix with random values
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')

    # Only tiny number of inputs per neuron so do the maths ourself rather than a big matrix and waste elements
    def forward(self, x: Tensor) -> Tensor:
        return (x * self.weight).sum(dim=-1) + self.bias
    
class DenseForward(nn.Linear):

    # Fully connected
    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)
    
class SparseLinearNeq(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        input_quant: QuantizerBase,
        output_quant: QuantizerBase,
        imask: Tensor,
        dense_forward: bool,
        fan_in: int,
        width_n: int,
        apply_input_quant: bool = True,
        apply_output_quant: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_quant = input_quant
        self.output_quant = output_quant
        self.input_quant.float_output()
        self.output_quant.float_output()
        self.imask = imask
        self.dense_forward = dense_forward
        self.fan_in = fan_in
        self.width_n = width_n
        self.apply_input_quant = apply_input_quant
        self.apply_output_quant = apply_output_quant

        self.relu = nn.ReLU(inplace=True)
        self.fc1 = SparseLinear(fan_in, out_features * width_n)
        self.fc4 = SparseLinear(width_n, out_features)
        self.res2 = SparseLinear(fan_in, out_features)
        self.fc1_dense = DenseForward(in_features, out_features * width_n)
        self.res2_dense = DenseForward(in_features, out_features)

    def forward(self, x: Tensor) -> Tensor:
        if (self.dense_forward):
            return self._dense_forward(x)

        return self._sparse_forward(x)
    
    def _sparse_forward(self, x: Tensor) -> Tensor:
        if self.apply_input_quant:
            x = self.input_quant(x)

        # [B, in_features] -> [B, out_features, fan_in] - select fan_in inputs per neuron via imask
        # x[0] = [a1,b1,c1,d1,...], imask=[[0,3],[1,4]] -> [[[a1,d1],[b1,e1]], [[a2,d2],[b2,e2]]]
        x = x[:, self.imask]
        residual = self.res2(x)

        # Copy each input width_n times so now shape is [batch, out_features*width_n, fan_in]
        # [[[a1,d1], [a1,d1], [b1,e1], [b1,e1]],
        # [[a2,d2], [a2,d2], [b2,e2], [b2,e2]]]
        x = x.repeat(1, 1, self.width_n).reshape(x.size(0), x.size(1)*self.width_n, self.fan_in)
        x = self.fc1(x)
        x = self.relu(x)

        # Now width_n inputs, and we have out_features number of neurons
        # [[[fc1_n0_w0, fc1_n0_w1], [fc1_n1_w0, fc1_n1_w1]],
        # [[fc1_n0_w0, fc1_n0_w1], [fc1_n1_w0, fc1_n1_w1]]]

        # Now each neuron has its width_n fc1 outputs grouped together, 
        # ready for fc4 to collapse [width_n] -> [1] per neuron, giving 
        # final shape (B, out_features)
        x = x.reshape(x.size(0), x.size(1) // self.width_n, self.width_n)
        x = self.fc4(x)
        
        x = x + residual

        if self.apply_output_quant:
            x = self.output_quant(x)
        return x

    def _dense_forward(self, x: Tensor) -> Tensor:
        if self.apply_input_quant:
            x = self.input_quant(x)

        # No mask as dense
        residual = self.res2_dense(x)

        # No need to implicitly duplicate all inputs as matrix multiply
        x = self.fc1_dense(x)
        x = self.relu(x)

        # Had width_n*out_feature outputs from fc1, now grouped into out_features
        # number of width_n's for fc4
        x = x.reshape(x.size(0), x.size(1) // self.width_n, self.width_n)
        x = self.fc4(x)

        # Both of size out_features
        x = x + residual

        if self.apply_output_quant:
            x = self.output_quant(x)
        return x

class LUTLayer(nn.Module):
    def __init__(self, neq: SparseLinearNeq):
        super().__init__()
        self.neq = neq
        self.neuron_truth_tables = None
        self.neq.input_quant.bin_output()
        self.neq.output_quant.bin_output()

    def calculate_truth_tables(self) -> None:
        # Don't use anything we do in here for training
        with torch.no_grad():
            float_state_space = self.neq.input_quant.state_space(as_float = True)
            int_state_space = self.neq.input_quant.state_space(as_float = False)
            
            float_perm = generate_permutation_matrix([float_state_space] * self.neq.fan_in)
            int_perm   = generate_permutation_matrix([int_state_space] * self.neq.fan_in)

            # Run the tree once in float mode for human-readable output states
            self.neq.output_quant.float_output()
            float_out = self.neq.output_quant(
                self.forward_to_fill_luts(float_perm)
            )

            # Run again in bin mode for the integer addresses stored in the table
            self.neq.output_quant.bin_output()
            bin_out = self.neq.output_quant(
                self.forward_to_fill_luts(float_perm)
            )

                # Reset quant to float mode for any further training/eval
            self.neq.output_quant.float_output()

            # Store one tuple per output neuron
            self.neuron_truth_tables = [
                (
                    self.neq.imask[n],   # which input indices this neuron connects to
                    int_perm,            # [num_combos, fan_in] integer input addresses
                    float_out[:, n],     # [num_combos] float outputs
                    bin_out[:, n],       # [num_combos] integer outputs
                )
                for n in range(self.neq.out_features)
            ]

    # Look up inference using the truth tables
    def forward(self, x: Tensor) -> Tensor:
        assert self.neuron_truth_tables is not None, "Call calculate_truth_tables() first"

        self.neq.input_quant.bin_output()
        
        x = self.neq.input_quant(x)

        y = torch.zeros(x.size(0), self.neq.out_features)  # [B, out_features]
        for i in range(self.neq.out_features):
            indices, int_perm, _, bin_out = self.neuron_truth_tables[i]
            # x[:, indices]: slice this neuron's fan_in inputs from full batch -> [B, fan_in]
            # _table_lookup: find matching combo row, return stored integer output -> [B]
            y[:, i] = self._table_lookup(x[:, indices], int_perm, bin_out)

        return y

    # Called by calculate_truth_tables() to fill the tree with outputs given an input tensor 
    # for every possible combination of inputs (our batch size)
    def forward_to_fill_luts(self, x: Tensor) -> Tensor:
        # Repeat permutation matrix for each outptu_feature to get a skip per output
        x = x.repeat(1, self.neq.out_features).reshape(x.size(0), self.neq.out_features, self.neq.fan_in)
        residual = self.neq.res2(x)

        # Repeat per width_n and reshape it
        x = x.repeat(1, 1, self.neq.width_n).reshape(x.size(0), x.size(1) * self.neq.width_n, self.neq.fan_in)
        x = self.neq.fc1(x)
        x = self.neq.relu(x)

        # Had width_n*out_feature outputs from fc1, now grouped into out_features
        # number of width_n's for fc4
        x = x.reshape(x.size(0), x.size(1) // self.neq.width_n, self.neq.width_n)
        x = self.neq.fc4(x)

        # Both of size out_features
        x = x + residual

        # No post transform as they are all float based
        return x
    
    # Look up inference using the truth tables
    def _table_lookup(
        self,
        connected_input: Tensor,
        int_perm: Tensor,
        bin_out: Tensor,
    ) -> Tensor:
        # [B, fan_in] -> [B, fan_in, 1] - add dim to broadcast against all combos
        ci = connected_input.unsqueeze(2)
        # [num_combos, fan_in] -> [1, fan_in, num_combos] - add batch dim, transpose so fan_in aligns
        pm = int_perm.t().unsqueeze(0)
        # broadcast [B, fan_in, 1] == [1, fan_in, num_combos] -> [B, fan_in, num_combos]
        # sum over fan_in: how many positions matched per (sample, combo) pair -> [B, num_combos]
        # == fan_in: True only where ALL fan_in positions matched i.e. full input vector matches combo
        eq = (ci == pm).sum(dim=1) == self.neq.fan_in

        # each sample must match exactly one combo - sanity check quantization is correct
        matches = eq.sum(dim=1)
        if not (matches == torch.ones_like(matches, dtype=matches.dtype)).all():
            raise Exception("One or more vectors in the input is not in the possible input state space")

        # index of the matching combo row per sample -> [B]
        idx = torch.argmax(eq.to(torch.int64), dim=1)
        # return the stored integer output for each sample's matching combo
        return bin_out[idx]

# Tuples inputs into pairs 
def SupportMask(out_features: int, fan_in: int):
    imask = torch.arange(out_features).reshape([out_features//fan_in,fan_in])
    return imask

class ScalarScaleBias(nn.Module):
    def __init__(self, scale=True, scale_init=1.0, bias=True, bias_init=0.0) -> None:
        super(ScalarScaleBias, self).__init__()
        if scale:
            self.weight = Parameter(torch.Tensor(1))
        else:
            self.register_parameter("weight", None)
        if bias:
            self.bias = Parameter(torch.Tensor(1))
        else:
            self.register_parameter("bias", None)
        self.weight_init = scale_init
        self.bias_init = bias_init
        self.reset_parameters()

    # Change the default initialisation for BatchNorm
    def reset_parameters(self) -> None:
        if self.weight is not None:
            init.constant_(self.weight, self.weight_init)
        if self.bias is not None:
            init.constant_(self.bias, self.bias_init)

    def forward(self, x):
        if self.weight is not None:
            x = x * self.weight
        if self.bias is not None:
            x = x + self.bias
        return x


class ScalarBiasScale(ScalarScaleBias):
    def forward(self, x):
        if self.bias is not None:
            x = x + self.bias
        if self.weight is not None:
            x = x * self.weight
        return x
