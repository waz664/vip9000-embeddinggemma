import onnx
import numpy as np
from onnx import helper, numpy_helper, TensorProto
from collections import Counter

src = "embeddinggemma_seq128_bias_hidden.onnx"
dst = "embeddinggemma_seq128_bias_hidden_npu.onnx"
m = onnx.load(src)
constants = {}
for init in m.graph.initializer:
    constants[init.name] = numpy_helper.to_array(init)
for node in m.graph.node:
    for attr in node.attribute:
        if attr.HasField("t"):
            constants[node.output[0]] = numpy_helper.to_array(attr.t)
            break

new_nodes = []
new_initializers = list(m.graph.initializer)
expand_replaced = 0
gelus = 0
for node in m.graph.node:
    if node.op_type == "Expand":
        input_name, shape_name = node.input[0], node.input[1]
        output_name = node.output[0]
        if shape_name not in constants:
            raise RuntimeError(f"Expand shape not constant for {node.name or output_name}: {shape_name}")
        target_shape = constants[shape_name].astype(np.int64)
        ones_name = f"ones_for_{output_name}"
        new_initializers.append(numpy_helper.from_array(np.ones(target_shape, dtype=np.float32), name=ones_name))
        new_nodes.append(helper.make_node("Mul", [input_name, ones_name], [output_name], name=(node.name or output_name) + "_mulones"))
        expand_replaced += 1
        continue
    if node.op_type == "Gelu":
        x, y = node.input[0], node.output[0]
        base = (node.name or y).replace("/", "_").replace(":", "_") + "_gelu"
        def const(name, val):
            return helper.make_node("Constant", [], [name], value=helper.make_tensor(name + "_t", TensorProto.FLOAT, [], [val]))
        c_half = base + "_half"; c_one = base + "_one"; c_alpha = base + "_alpha"; c_beta = base + "_beta"
        x2 = base + "_x2"; x3 = base + "_x3"; beta_x3 = base + "_beta_x3"; inner = base + "_inner"
        scaled = base + "_scaled"; th = base + "_tanh"; plus = base + "_plus"; xh = base + "_xhalf"
        new_nodes.extend([
            const(c_half, 0.5), const(c_one, 1.0), const(c_alpha, 0.7978845608028654), const(c_beta, 0.044715),
            helper.make_node("Mul", [x, x], [x2], name=base + "_x2"),
            helper.make_node("Mul", [x2, x], [x3], name=base + "_x3"),
            helper.make_node("Mul", [x3, c_beta], [beta_x3], name=base + "_beta"),
            helper.make_node("Add", [x, beta_x3], [inner], name=base + "_inner"),
            helper.make_node("Mul", [inner, c_alpha], [scaled], name=base + "_scaled"),
            helper.make_node("Tanh", [scaled], [th], name=base + "_tanh"),
            helper.make_node("Add", [th, c_one], [plus], name=base + "_plus"),
            helper.make_node("Mul", [x, c_half], [xh], name=base + "_xhalf"),
            helper.make_node("Mul", [xh, plus], [y], name=base + "_out"),
        ])
        gelus += 1
        continue
    new_nodes.append(node)

m.graph.ClearField("node")
m.graph.node.extend(new_nodes)
m.graph.ClearField("initializer")
m.graph.initializer.extend(new_initializers)
for inp in m.graph.input:
    if inp.name == "inputs_embeds":
        t = inp.type.tensor_type; t.elem_type = TensorProto.FLOAT; t.shape.ClearField("dim")
        for size in [1,128,768]:
            d = t.shape.dim.add(); d.dim_value = size
    if inp.name == "attention_bias":
        t = inp.type.tensor_type; t.elem_type = TensorProto.FLOAT; t.shape.ClearField("dim")
        for size in [1,1,1,128]:
            d = t.shape.dim.add(); d.dim_value = size
onnx.checker.check_model(m)
onnx.save(m, dst)
print("saved", dst, "expand", expand_replaced, "gelu", gelus)
print(Counter(n.op_type for n in m.graph.node).most_common(50))
