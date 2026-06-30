from .nn import LUTLayer
from .verilog_templates import *
from .bench import BenchGenerator

class VerilogGenerator:
    def __init__(self, layer: LUTLayer) -> None:
        assert layer.neuron_truth_tables is not None, "Call calculate_truth_tables() first"
        self.layer = layer

    def generate(self, prefix: str, directory: str) -> tuple[int, int]:
        _, input_bitwidth  = self.layer.neq.input_quant.get_scale_factor_bits()
        _, output_bitwidth = self.layer.neq.output_quant.get_scale_factor_bits()
        input_bitwidth  = int(input_bitwidth)
        output_bitwidth = int(output_bitwidth)
        total_input_bits  = self.layer.neq.in_features  * input_bitwidth
        total_output_bits = self.layer.neq.out_features * output_bitwidth

        # Write one .v file per neuron
        for index in range(self.layer.neq.out_features):
            module_name = f"{prefix}_N{index}"
            with open(f"{directory}/{module_name}.v", "w") as f:
                f.write(self._gen_neuron(index, module_name, input_bitwidth, output_bitwidth))

        # Write the layer wrapper .v file
        with open(f"{directory}/{prefix}.v", "w") as f:
            f.write(self._gen_layer(prefix, input_bitwidth, output_bitwidth))

        return total_input_bits, total_output_bits
    
    def _gen_layer(self, prefix: str, input_bw: int, output_bw: int) -> str:
        total_in  = self.layer.neq.in_features  * input_bw
        total_out = self.layer.neq.out_features * output_bw
        contents  = f"module {prefix} (input [{total_in-1}:0] M0, output [{total_out-1}:0] M1);\n\n"
        output_offset = 0
        for index in range(self.layer.neq.out_features):
            module_name = f"{prefix}_N{index}"
            indices, _, _, _ = self.layer.neuron_truth_tables[index]
            connection = generate_neuron_connection_verilog(indices, input_bw)
            wire_name  = f"{module_name}_wire"
            contents  += f"wire [{len(indices)*input_bw-1}:0] {wire_name} = {{{connection}}};\n"
            contents  += f"{module_name} {module_name}_inst (.M0({wire_name}), .M1(M1[{output_offset+output_bw-1}:{output_offset}]));\n\n"
            output_offset += output_bw
        contents += "endmodule"
        return contents

    def _gen_neuron(self, index: int, module_name: str, input_bw: int, output_bw: int) -> str:
        indices, int_perm, _, bin_out = self.layer.neuron_truth_tables[index]
        cat_input_bw = len(indices) * input_bw
        lut_string   = ""
        for i in range(int_perm.shape[0]):
            entry = "".join(
                self.layer.neq.input_quant.to_bin_str(int_perm[i, j].item())
                for j in range(len(indices))
            )
            result     = self.layer.neq.output_quant.to_bin_str(bin_out[i].item())
            lut_string += f"\t\t\t{cat_input_bw}'b{entry}: M1r = {output_bw}'b{result};\n"
        return generate_lut_verilog(module_name, cat_input_bw, output_bw, lut_string)
    
# Function to generate a Verilog module that connects multiple LUTLayer modules in sequence, with optional registers and bench file generation
def module_list_to_verilog_module(
    module_list,
    module_name: str,
    output_directory: str,
    add_registers: bool = False,
    generate_bench: bool = False,
) -> None:

    input_bitwidth = None
    output_bitwidth = None
    module_contents = ""

    for i, layer in enumerate(module_list):
        if not isinstance(layer, LUTLayer):
            raise TypeError(f"Expected LUTLayer, got {type(layer)}")
        prefix = f"layer{i}"
        vgen = VerilogGenerator(layer)
        layer_input_bits, layer_output_bits = vgen.generate(prefix, output_directory)

        if i == 0:
            input_bitwidth = layer_input_bits
        if i == len(module_list) - 1:
            output_bitwidth = layer_output_bits

        module_contents += layer_connection_verilog(
            layer_string=prefix,
            input_string=f"M{i}",
            input_bits=layer_input_bits,
            output_string=f"M{i+1}",
            output_bits=layer_output_bits,
            output_wire=i != len(module_list) - 1,
            register=add_registers,
        )

        if generate_bench:
            bgen = BenchGenerator(layer)
            bgen.generate(prefix, output_directory)

    with open(f"{output_directory}/myreg.v", "w") as f:
        f.write(generate_register_verilog())

    with open(f"{output_directory}/{module_name}.v", "w") as f:
        f.write(generate_logicnets_verilog(
            module_name=module_name,
            input_name="M0",
            input_bits=input_bitwidth,
            output_name=f"M{len(module_list)}",
            output_bits=output_bitwidth,
            module_contents=module_contents,
        ))