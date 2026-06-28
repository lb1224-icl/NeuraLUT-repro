from functools import reduce

from .nn import LUTLayer

class BenchGenerator:
    def __init__(self, layer: LUTLayer) -> None:
        assert layer.neuron_truth_tables is not None, "Call calculate_truth_tables() first"
        self.layer = layer

    def generate(self, prefix: str, directory: str) -> None:
        for index in range(self.layer.neq.out_features):
            module_name = f"{prefix}_N{index}"
            with open(f"{directory}/{module_name}.bench", "w") as f:
                f.write(self._gen_neuron(index))

    def _gen_neuron(self, index: int) -> str:
        _, input_bw  = self.layer.neq.input_quant.get_scale_factor_bits()
        _, output_bw = self.layer.neq.output_quant.get_scale_factor_bits()
        input_bw, output_bw = int(input_bw), int(output_bw)

        indices, int_perm, _, bin_out = self.layer.neuron_truth_tables[index]
        cat_input_bw = len(indices) * input_bw
        num_entries  = int_perm.shape[0]

        # Convert each row's input integers to binary strings for sort_to_bench
        input_bin_strs = [
            [self.layer.neq.input_quant.to_bin_str(int_perm[i, j].item()) for j in range(len(indices))]
            for i in range(num_entries)
        ]
        sorted_bin_out = sort_to_bench(input_bin_strs, bin_out)

        lut_string = ""
        for bit in range(output_bw):
            output_bin_str = "".join(
                self.layer.neq.output_quant.to_bin_str(v.item())[output_bw - 1 - bit]
                for v in sorted_bin_out
            )
            hex_str     = f"{int(output_bin_str, 2):0{num_entries // 4}x}"
            lut_string += f"M1[{bit}]       = LUT 0x{hex_str} "
            lut_string += generate_lut_input_string(cat_input_bw)

        return generate_lut_bench(cat_input_bw, output_bw, lut_string)

def generate_lut_bench(input_fanin_bits, output_bits, lut_string):
    lut_neuron_template = """\
{input_string}\
{output_string}\
{lut_string}"""
    input_string = ""
    for i in range(input_fanin_bits):
        input_string += f"INPUT(M0[{i}])\n"
    output_string = ""
    for i in range(output_bits):
        output_string += f"OUTPUT(M1[{i}])\n"
    return lut_neuron_template.format(  input_string=input_string,
                                        output_string=output_string,
                                        lut_string=lut_string)

def generate_lut_input_string(input_fanin_bits):
    lut_input_string = ""
    for i in range(input_fanin_bits):
        if i == 0:
            lut_input_string += f"( M0[{i}]"
        elif i == input_fanin_bits-1:
            lut_input_string += f", M0[{i}] )\n"
        else:
            lut_input_string += f", M0[{i}]"
    return lut_input_string

def sort_to_bench(input_state_space_bin_str, bin_output_states):
    sorted_bin_output_states = bin_output_states.tolist()
    input_state_space_flat_int = list(map(lambda l: int(reduce(lambda a,b: a+b, l),2), input_state_space_bin_str))
    zipped_io_states = list(zip(input_state_space_flat_int, sorted_bin_output_states))
    zipped_io_states.sort(key=lambda x: x[0], reverse=True)
    sorted_bin_output_states = list(map(lambda x: x[1], zipped_io_states))
    return sorted_bin_output_states

