import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper, numpy_helper
import numpy as np


def make_const(name, values, dtype=np.int64):
    return numpy_helper.from_array(np.asarray(values, dtype=dtype), name=name)


def get_tensor_shape(value_info):
    dims = []
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return None

    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)
        else:
            dims.append(None)
    return dims


def split_output_shapes(original_shape, axis):
    if original_shape is None:
        return None, None

    boxes_shape = list(original_shape)
    scores_shape = list(original_shape)
    axis = axis if axis >= 0 else len(original_shape) + axis

    if axis < 0 or axis >= len(original_shape):
        return None, None

    boxes_shape[axis] = 4
    scores_shape[axis] = 1
    return boxes_shape, scores_shape


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="input ONNX path with output shape like [1, 5, 8400]")
    parser.add_argument("--output", required=True, help="output ONNX path with split boxes/scores outputs")
    parser.add_argument("--axis", type=int, default=1, help="channel axis of YOLO output")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(input_path))
    graph = model.graph

    if len(graph.output) != 1:
        raise ValueError(f"Expected exactly one graph output, got {len(graph.output)}")

    original_output = graph.output[0]
    original_name = original_output.name
    original_shape = get_tensor_shape(original_output)
    if original_shape is None:
        original_shape = [1, 5, 8400]
    boxes_shape, scores_shape = split_output_shapes(original_shape, args.axis)

    consts = [
        make_const("split_boxes_starts", [0]),
        make_const("split_boxes_ends", [4]),
        make_const("split_scores_starts", [4]),
        make_const("split_scores_ends", [5]),
        make_const("split_axis", [args.axis]),
        make_const("split_steps", [1]),
    ]
    graph.initializer.extend(consts)

    boxes_name = original_name + "_boxes"
    scores_name = original_name + "_scores"

    graph.node.extend(
        [
            helper.make_node(
                "Slice",
                inputs=[original_name, "split_boxes_starts", "split_boxes_ends", "split_axis", "split_steps"],
                outputs=[boxes_name],
                name="split_output_boxes",
            ),
            helper.make_node(
                "Slice",
                inputs=[original_name, "split_scores_starts", "split_scores_ends", "split_axis", "split_steps"],
                outputs=[scores_name],
                name="split_output_scores",
            ),
        ]
    )

    del graph.output[:]
    graph.output.extend(
        [
            helper.make_tensor_value_info(boxes_name, TensorProto.FLOAT, boxes_shape),
            helper.make_tensor_value_info(scores_name, TensorProto.FLOAT, scores_shape),
        ]
    )

    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))
    print(f"Saved split-output ONNX: {output_path}")


if __name__ == "__main__":
    main()
