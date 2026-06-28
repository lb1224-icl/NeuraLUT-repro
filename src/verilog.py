from .nn import SparseLinearNeq
from .nn import LUTLayer

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
    
def generate_neuron_connection_verilog(input_indices, input_bitwidth):
    connection_string = ""
    for i in range(len(input_indices)):
        index = input_indices[i]
        offset = index*input_bitwidth
        for b in reversed(range(input_bitwidth)):
            connection_string += f"M0[{offset+b}]"
            if not (i == len(input_indices)-1 and b == 0):
                connection_string += ", "
    return connection_string

def generate_lut_verilog(module_name, input_fanin_bits, output_bits, lut_string):
    lut_neuron_template = """\
module {module_name} ( input [{input_fanin_bits_1:d}:0] M0, output [{output_bits_1:d}:0] M1 );

	(*rom_style = "distributed" *) reg [{output_bits_1:d}:0] M1r;
	assign M1 = M1r;
	always @ (M0) begin
		case (M0)
{lut_string}
		endcase
	end
endmodule\n"""
    return lut_neuron_template.format(  module_name=module_name,
                                        input_fanin_bits_1=input_fanin_bits-1,
                                        output_bits_1=output_bits-1,
                                        lut_string=lut_string)

def layer_connection_verilog(layer_string: str, input_string: str, input_bits: int, output_string: str, output_bits: int, output_wire=True, register=False):
    if register:
        layer_connection_template = """\
wire [{input_bits_1:d}:0] {input_string}w;
myreg #(.DataWidth({input_bits})) {layer_string}_reg (.data_in({input_string}), .clk(clk), .rst(rst), .data_out({input_string}w));\n"""
    else:
        layer_connection_template = """\
wire [{input_bits_1:d}:0] {input_string}w;
assign {input_string}w = {input_string};\n"""
    layer_connection_template += "wire [{output_bits_1:d}:0] {output_string};\n" if output_wire else ""
    layer_connection_template += "{layer_string} {layer_string}_inst (.M0({input_string}w), .M1({output_string}));\n"
    return layer_connection_template.format(    layer_string=layer_string,
                                                input_string=input_string,
                                                input_bits=input_bits,
                                                input_bits_1=input_bits-1,
                                                output_string=output_string,
                                                output_bits_1=output_bits-1)