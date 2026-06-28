import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch import Tensor
from torch.nn import init
from .quant import QuantizerBase
from .util import generate_permutation_matrix

class SparseLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        super(SparseLinear, self).__init__(
            in_features=in_features, out_features=out_features, bias=bias
        )
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')

    def forward(self, input: Tensor) -> Tensor:
        return (input * self.weight).sum(dim=-1) + self.bias

class DenseForward(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(DenseForward, self).__init__(in_features=in_features, out_features=out_features, bias=bias)

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
        support: bool,
        dense_forward: bool,
        fan_in: int,
        width_n: int,
        apply_input_quant: bool = True,
        apply_output_quant: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_quant = input_quant
        self.output_quant = output_quant
        self.input_quant.float_output()
        self.output_quant.float_output()
        self.imask = imask
        self.support = support
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
        if self.dense_forward:
            return self._dense_forward(x)
        return self._sparse_forward(x)

    def _dense_forward(self, x: Tensor) -> Tensor:
        if self.apply_input_quant:
            x = self.input_quant(x)
        residual = self.res2_dense(x)
        x = self.fc1_dense(x)
        x = self.relu(x)
        x = x.reshape(x.size(0), x.size(1) // self.width_n, self.width_n)
        x = self.fc4(x)
        x = x + residual
        if self.apply_output_quant:
            x = self.output_quant(x)
        return x

    def _sparse_forward(self, x: Tensor) -> Tensor:
        if self.apply_input_quant:
            x = self.input_quant(x)
        x = x[:, self.imask]
        residual = self.res2(x)
        x = x.repeat(1, 1, self.width_n).reshape(x.size(0), x.size(1) * self.width_n, self.fan_in)
        x = self.fc1(x)
        x = self.relu(x)
        x = x.reshape(x.size(0), x.size(1) // self.width_n, self.width_n)
        x = self.fc4(x)
        x = x + residual
        if self.apply_output_quant:
            x = self.output_quant(x)
        return x
    
class LUTLayer(nn.Module):
    def __init__(self, neq: SparseLinearNeq) -> None:
        super().__init__()
        self.neq = neq
        self.neuron_truth_tables = None
        self.neq.input_quant.bin_output()
        self.neq.output_quant.bin_output()

    def calculate_truth_tables(self) -> None:
        with torch.no_grad():
            # All representable input values (floats for the tree, ints for the table)
            float_state_space = self.neq.input_quant.state_space(as_float=True)
            int_state_space   = self.neq.input_quant.state_space(as_float=False)

            # Build [num_combos, fan_in] matrix of every possible input combination
            float_perm = generate_permutation_matrix([float_state_space] * self.neq.fan_in)
            int_perm   = generate_permutation_matrix([int_state_space]   * self.neq.fan_in)

            # Run the tree once in float mode for human-readable output states
            self.neq.output_quant.float_output()
            float_out = self.neq.output_quant(
                self.neq.forward_to_fill_luts(float_perm)
            )

            # Run again in bin mode for the integer addresses stored in the table
            self.neq.output_quant.bin_output()
            bin_out = self.neq.output_quant(
                self.neq.forward_to_fill_luts(float_perm)
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

    def forward(self, x: Tensor) -> Tensor:
        assert self.neuron_truth_tables is not None, "Call calculate_truth_tables() first"
        self.neq.input_quant.bin_output()
        x = self.neq.input_quant(x)
        y = torch.zeros(x.shape[0], self.neq.out_features)
        for i in range(self.neq.out_features):
            indices, int_perm, _, bin_out = self.neuron_truth_tables[i]
            y[:, i] = self._table_lookup(x[:, indices], int_perm, bin_out)
        return y
    
    def forward_to_fill_luts(self, x: Tensor) -> Tensor:
        # x is [num_combos, fan_in] — already shaped per neuron, no imask needed
        x = x.repeat(1, self.neq.out_features).reshape(x.shape[0], self.neq.out_features, self.neq.fan_in)
        residual = self.neq.res2(x)
        x = x.repeat(1, 1, self.neq.width_n).reshape(x.size(0), x.size(1) * self.neq.width_n, self.neq.fan_in)
        x = self.neq.fc1(x)
        x = self.neq.relu(x)
        x = x.reshape(x.size(0), x.size(1) // self.neq.width_n, self.neq.width_n)
        x = self.neq.fc4(x)
        x = x + residual
        return x

    def _table_lookup(
        self,
        connected_input: Tensor,
        int_perm: Tensor,
        bin_out: Tensor,
    ) -> Tensor:
        ci  = connected_input.unsqueeze(2)       # [B, fan_in, 1]
        pm  = int_perm.t().unsqueeze(0)           # [1, fan_in, num_combos]
        eq  = (ci == pm).sum(dim=1) == self.neq.fan_in  # [B, num_combos]

        matches = eq.sum(dim=1)  # Count the number of perfect matches per input vector
        if not (matches == torch.ones_like(matches, dtype=matches.dtype)).all():
            raise Exception(
                f"One or more vectors in the input is not in the possible input state space"
            )

        idx = torch.argmax(eq.to(torch.int64), dim=1)
        return bin_out[idx]
    
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
