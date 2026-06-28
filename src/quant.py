import torch
import torch.nn as nn
from torch import Tensor

from brevitas.core.quant import QuantType

from abc import ABC, abstractmethod

class QuantizerBase(nn.Module, ABC):
    def __init__(self, pre_transforms=None, post_transforms=None):
        super().__init__()
        self.is_bin_output = False
        self.pre_transforms = nn.ModuleList(pre_transforms or [])
        self.post_transforms = nn.ModuleList(post_transforms or [])

    # Get the scale factor and the number of bits for quantization
    @abstractmethod
    def get_scale_factor_bits(self) -> tuple:
        pass

    # Returns a tensor of every integer/floating-point value representable in N bits
    @abstractmethod
    def state_space(self, as_float: bool = True) -> Tensor: 
        pass

    # Returns the binary string representation of an integer value
    @abstractmethod
    def to_bin_str(self, x: int) -> str:
        pass

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass

    # Set the output to be binary instead of float
    def bin_output(self) -> bool:
        self.is_bin_output = True
        return self.is_bin_output
    
    # Set the output to be float instead of binary
    def float_output(self) -> bool:
        self.is_bin_output = False
        return not self.is_bin_output
    
    def apply_pre_transforms(self, x: Tensor) -> Tensor:
        for t in self.pre_transforms:
            x = t(x)
        return x

    def apply_post_transforms(self, x: Tensor) -> Tensor:
        for t in self.post_transforms:
            x = t(x)
        return x
    
    def _unsupported_quant_error(self) -> NotImplementedError:
        return NotImplementedError(f"Quantisation type {self.quant_type} not supported")

class BrevitasQuantizedActivation(QuantizerBase):
    def __init__(self, brevitas_module, quant_type, bit_width, 
                 pre_transforms=None, post_transforms=None, cuda=False):
        super().__init__(pre_transforms, post_transforms)
        self.brevitas_module = brevitas_module
        self.quant_type = quant_type      # passed explicitly, not inferred
        self.bit_width = bit_width
        self.narrow_range = self.brevitas_module.narrow_range
        self.cuda = cuda
        if(self.cuda):
            self.brevitas_module = self.brevitas_module.cuda()
    
    def get_scale_factor_bits(self) -> tuple:
        if self.quant_type != QuantType.INT:
            raise self._unsupported_quant_error()

        scale_factor = self.brevitas_module.scale.detach().cpu().numpy()
        bit_width = self.bit_width

        return scale_factor, bit_width

    def get_range(self) -> tuple[int, int]:
        if self.quant_type != QuantType.INT:
            raise self._unsupported_quant_error()
        
        _, bit_width = self.get_scale_factor_bits()

        min_val = -(2 ** (bit_width - 1)) + int(self.narrow_range)
        max_val = (2 ** (bit_width - 1)) - 1

        return min_val, max_val

    def state_space(self, as_float: bool = True) -> Tensor:
        min_val, max_val = self.get_range()
        integers = torch.arange(min_val, max_val + 1, dtype=torch.int32)
        if as_float:
            scale_factor, _ = self.get_scale_factor_bits()
            return self.apply_post_transforms(integers.float() * float(scale_factor))
        return integers
    
    def to_bin_str(self, x: int) -> str:
        if self.quant_type != QuantType.INT:
            raise self._unsupported_quant_error()
        
        _, bit_width = self.get_scale_factor_bits()
        offset = 2 ** (bit_width - 1) - int(self.narrow_range) # ensures lowest representable integer is 0 for all ranges

        return format(int(x) + offset, f'0{bit_width}b')
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.apply_pre_transforms(x)
        x = self.brevitas_module(x)
        if self.is_bin_output:
            scale_factor, _ = self.get_scale_factor_bits()
            return torch.round(x / float(scale_factor)).to(torch.int64)
        return self.apply_post_transforms(x)

    def get_quant_type(self) -> QuantType:
        return self.quant_type
        
    
