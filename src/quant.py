import torch
import torch.nn as nn
from torch import Tensor

from brevitas.core.quant import QuantType

from abc import ABC, abstractmethod

class QuantizerBase(nn.Module, ABC):
    def __init__(self, pre_transforms = None, post_transforms=None):
        super().__init__()
        self.is_bin_output = False
        self.pre_transforms = pre_transforms
        self.post_transforms = post_transforms  

    @abstractmethod
    def forward(self, x:Tensor) -> Tensor:
        pass

    # Returns a binary string equivalent of integer
    @abstractmethod
    def to_bin_str(self, x:int) -> str:
        pass

    # Returns a tensor of every integer/floating-point value representable in N bits
    @abstractmethod
    def state_space(self, as_float: bool) -> Tensor:
        pass

    # Returns a tuple of scale factor and number of bits for quantisation
    @abstractmethod
    def get_scale_factor_bits(self) -> tuple:
        pass
    
    def apply_pre_transforms(self, x: Tensor) -> Tensor:
        for pre in self.pre_transforms:
            x = pre(x)
        return x

    def apply_post_transforms(self, x: Tensor) -> Tensor:
        for post in self.post_transforms:
            x = post(x)
        return x

    # Set the output to be binary instead of float
    def bin_output(self) -> bool:
        self.is_bin_output = True
        return self.is_bin_output
    
    # Set the output to be float instead of binary
    def float_output(self) -> bool:
        self.is_bin_output = False
        return not self.is_bin_output
    
    def _unsupported_quant_error(self) -> NotImplementedError:
        return NotImplementedError(f"Quantisation type {self.quant_type} not supported")

class BrevitasQuantizedActivation(QuantizerBase):
    def __init__(self, brevitas_module, quant_type, bit_width, narrow_range, cuda = False):
        super().__init__()
        self.brevitas_module = brevitas_module
        self.quant_type = quant_type
        self.bit_width = bit_width
        self.narrow_range = narrow_range
        self.cuda = cuda
        if(self.cuda): self.brevitas_module.cuda

    def get_scale_factor_bits(self) -> tuple:
        if self.quant_type != QuantType.INT:
            raise self._unsupported_quant_error()
        
        scale_factor = self.brevitas_module.act_quant.scale().detach().cpu().numpy()
        
        return scale_factor, self.bit_width
    
    # Returns the integer range from bit_width pre transforms or scaling
    def get_range(self) -> tuple[int, int]:
        if(self.quant_type != QuantType.INT):
            raise self._unsupported_quant_error()
        
        _, bit_width = self.get_scale_factor_bits()
        min_val = -(2 ** (bit_width-1)) + int(self.narrow_range)
        max_val = (2 ** (bit_width-1))

        return (min_val, max_val)
    
    # Used to generate the truth tables, not in training
    def state_space(self, as_float: bool) -> Tensor:
        min_val, max_val = self.get_range()
        integers = torch.arange(min_val, max_val + 1, dtype=torch.int32)
        
        if (as_float):
            scale_factor, _ = self.get_scale_factor_bits()

            # applies scalarscalebias and scalarbiasscale
            return self.apply_post_transforms(integers.float() * float(scale_factor)) 

        return integers
    
    def to_bin_str(self, x:int) -> str:
        if(self.quant_type != QuantType.INT):
            raise self._unsupported_quant_error()
        
        _, bit_width = self.get_scale_factor_bits()

        offset = 2**(bit_width - 1) - int(self.narrow_range)

        return format(x + offset, f'{bit_width}b')
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.apply_pre_transforms(x)
        x = self.brevitas_module(x)
        if self.is_bin_output:
            scale_factor, _ = self.get_scale_factor_bits()
            return torch.round(x / float(scale_factor)).to(torch.int64)
        return self.apply_post_transforms(x)


    def get_quant_type(self) -> QuantType:
        return self.quant_type
